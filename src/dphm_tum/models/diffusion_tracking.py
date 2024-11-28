import pytorch3d.transforms
import trimesh
import torch
from torch import optim
import numpy as np
from pytorch3d.ops import knn_points, knn_gather
import pyvista as pv
from torch.autograd import grad
import torch.nn.functional as F
import wandb
#from dphm_tum.models.diff_operators import gradient, nabla
from dphm_tum.utils.landmark import lables

def inference_identity_fitting(neural3DMM,
                        diffusion_shape, diffusion_expr,
                        lats_exp, lat_shape, 
                        lambdas, sds_args,  all_points_exp, all_normals_exp, 
                        all_rotation_params, all_translation_params, all_scale_params, all_lms_exp, 
                        anchor_indices=(0, 1, 10, 11, 12, 13, 34, 35), init_lr=0.0002, init_lr_rigid=0.000001, n_steps=31, 
                        schedule_cfg={}, schedule_rigid_cfg={},
                        calculate_normal=False, calculate_anchor=False, opt_rigid_scale=False, opt_iden=True, opt_expre=True,
                        log_loss=False, project="fit_pointclouds_seq", exp_name="nphm_kinect",
                        use_quaternion = False, convention = 'ZYX', flame2scan = False):
    
    if log_loss:
        wandb.init(project=project, config=lambdas, name=exp_name)
        
    t_range = sds_args.get('t_range', [0.02, 0.98])
    guidance_scale = sds_args.get('guidance_scale', 100) 
    grad_scale = sds_args.get('grad_scale', 1) 
    skip_unet = sds_args.get('skip_unet', False)

    decoder_id = neural3DMM.id_model #identity_model
    decoder_exp = neural3DMM.ex_model #expression_model
    
    # nonrigid parameters
    if opt_iden:
        lat_shape.requires_grad = True
        optimizer_id = optim.Adam(params=[lat_shape], lr=init_lr) 
    if opt_expre:
        lats_exp.requires_grad = True
        optimizer_exp = optim.Adam(params=[lats_exp], lr=init_lr) 
        
    scale = torch.from_numpy(all_scale_params[0]).to(lat_shape.device)
    rotation =  torch.stack(all_rotation_params, axis=0).to(lat_shape.device)
    translation = torch.stack(all_translation_params, axis=0).to(lat_shape.device)
    if opt_rigid_scale:   
        print('optimize rigid scale together for better alignment !!!!!!!!!!!!!!!!!!')  
        scale.requires_grad = True
        rotation.requires_grad = True
        translation.requires_grad = True
        optimizer_rigid = optim.Adam(params=[rotation, translation, scale], lr=init_lr_rigid)

    all_points_exp_clone = []
    for i, points_exp in enumerate(all_points_exp):
        all_points_exp_clone.append(points_exp.detach().clone().unsqueeze(0))
        assert torch.isnan(points_exp).sum() == 0
    all_normals_exp_clone = []
    for i, normals_exp in enumerate(all_normals_exp):
        all_normals_exp_clone.append(normals_exp.detach().clone().unsqueeze(0))
        assert torch.isnan(normals_exp).sum() == 0


    decoder_id = neural3DMM.id_model #identity_model
    decoder_exp = neural3DMM.ex_model #expression_model

    all_frames = list(range(len(all_points_exp)))
    
    for j in range(n_steps):
        if int(j) in schedule_cfg['lr']:
            if opt_iden:
                for param_group in optimizer_id.param_groups:
                    param_group["lr"] /= schedule_cfg['lr'][int(j)]
                print('reduced LR for id')
            if opt_expre:
                for param_group in optimizer_exp.param_groups:
                    param_group["lr"] /= schedule_cfg['lr'][int(j)]
                print('reduced LR for expr')

        if opt_rigid_scale:
            if int(j) in schedule_rigid_cfg['lr']:
                for param_group in optimizer_rigid.param_groups:
                    param_group["lr"] /= schedule_rigid_cfg['lr'][int(j)]
                print('reduced LR for rigid')
        
        if opt_iden:
            optimizer_id.zero_grad()
        if opt_expre:
            optimizer_exp.zero_grad()
        if opt_rigid_scale:
            optimizer_rigid.zero_grad()

        loss_dict = {}                   

        # identity code regularization
        loss_dict['reg_glob'] = (torch.norm(lat_shape[..., :decoder_id.lat_dim_glob], dim=-1) ** 2).mean() 
        loss_dict['reg_loc'] = ((torch.norm(lat_shape[..., decoder_id.lat_dim_glob:], dim=-1) ** 2)).mean() 
        loc_lats_symm = lat_shape[:, :,
                        decoder_id.lat_dim_glob:decoder_id.lat_dim_glob + 2 * decoder_id.num_symm_pairs * decoder_id.lat_dim_loc].view(
            lat_shape.shape[0], decoder_id.num_symm_pairs * 2, decoder_id.lat_dim_loc)
        symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()
        symm_idx = [ (lables.index('eye_inner_left'), lables.index('eye_inner_right')),
                    (lables.index('eye_outer_left'), lables.index('eye_outer_right')),
                    (lables.index('mouth_corner_l'), lables.index('mouth_corner_r'))]
        symm_dist = 0
        for symm_pair in symm_idx:
            symm_dist += torch.norm(lat_shape[..., decoder_id.lat_dim_glob + symm_pair[0]*decoder_id.lat_dim_loc:decoder_id.lat_dim_glob + (symm_pair[0]+1)*decoder_id.lat_dim_loc] -
                                    lat_shape[..., decoder_id.lat_dim_glob + symm_pair[1]*decoder_id.lat_dim_loc:decoder_id.lat_dim_glob + (symm_pair[1]+1)*decoder_id.lat_dim_loc],
                                                dim=-1).square().mean() 
        loss_dict['symm_dist'] = symm_dist 

        extra_idx = [lables.index('lower_lip'), lables.index('mouth_corner_l'),
                    lables.index('mouth_corner_r')]  # + [10, 11, 12, 13]
        
        for idx in extra_idx:
            start, end = decoder_id.lat_dim_glob + idx * decoder_id.lat_dim_loc, decoder_id.lat_dim_glob + (idx + 1) * decoder_id.lat_dim_loc
            loss_dict['reg_extra'] = torch.norm(lat_shape[..., start:end], dim=-1).square().mean() 
        
        # calculate unconditional sds loss inspired by dreamfusion
        # forward_sds(img, t_range=[0.02, 0.98],  guidance_scale=100, grad_scale=1)
        min_step =  int(t_range[0] * diffusion_shape.num_timesteps)
        max_step =  int(t_range[1] * diffusion_shape.num_timesteps) + 1
        t = torch.randint(min_step, max_step, (lat_shape.size(0),), device=lat_shape.device).long()
        if 'sds_shape' in lambdas.keys():
            loss_dict['sds_shape'] = diffusion_shape.forward_sds(lat_shape, t=t,  guidance_scale=guidance_scale, grad_scale=grad_scale, skip_unet=skip_unet) 
        
        loss_dict['smooth_expr'] = (lats_exp[:-1] - lats_exp[1:]).norm(dim=-1).square().mean() 
        loss_dict['smooth_rot'] = (rotation[:-1] - rotation[1:]).norm(dim=-1).mean() 
        loss_dict['smooth_trans'] = (translation[:-1] - translation[1:]).norm(dim=-1).mean() 

        sampled_points = []
        sampled_normals = []
        sampled_landmarks = []
        for cc, c in enumerate(range(len(all_frames))):
            # more expression points
            n_samps =  10000
            if len(all_frames) >=6:
                n_samps = 5000
            subsample_idx = torch.randint(0, all_points_exp_clone[c].shape[1], [n_samps])
            p_exp = all_points_exp_clone[c][:, subsample_idx, :]
            n_exp = all_normals_exp_clone[c][:, subsample_idx, :]
            if flame2scan:
                if use_quaternion:
                    quan_or_euler_to_rot = pytorch3d.transforms.quaternion_to_matrix(rotation[c])
                else:
                    quan_or_euler_to_rot = pytorch3d.transforms.euler_angles_to_matrix(rotation[c], convention=convention)
                p_exp_nphm = ( (p_exp.squeeze() - translation[c].unsqueeze(0)) @ quan_or_euler_to_rot / scale ).unsqueeze(0)  * 4.0
                n_exp_nphm = (n_exp.squeeze() @ quan_or_euler_to_rot ).unsqueeze(0)
                lms_exp_nphm =  ( (all_lms_exp[c].squeeze() - translation[c].unsqueeze(0)) @ quan_or_euler_to_rot / scale ).unsqueeze(0)  * 4.0
            else:
                raise NotImplementedError
            #print(translation[c].shape, pytorch3d.transforms.euler_angles_to_matrix(rotation[c], convention=convention).shape, scale.shape)
            #print(p_exp_nphm.shape, n_exp_nphm.shape, lms_exp_nphm.shape)
            sampled_points.append(p_exp_nphm)
            sampled_normals.append(n_exp_nphm)
            sampled_landmarks.append(lms_exp_nphm)
        sampled_points = torch.concat(sampled_points, dim=0)
        sampled_normals = torch.concat(sampled_normals, dim=0)
        sampled_landmarks = torch.concat(sampled_landmarks, dim=0)

        #print(sampled_points.shape, lats_exp.shape, lat_shape.shape)
        #previous implementation:   
        #sdf = neural3DMM(sampled_points, lats_exp, lat_shape.repeat(lats_exp.shape[0], 1, 1))['sdf']
        #_, sdf_grad = nabla(neural3DMM, sampled_points, lats_exp, lat_shape.repeat(lats_exp.shape[0], 1, 1))
        out_dict = neural3DMM({'queries': sampled_points}, {'geo': lat_shape.repeat(lats_exp.shape[0], 1, 1), 'exp': lats_exp}, skip_color=True, ignore_deformations=False, return_grad=True)
        sdf, sdf_grad = out_dict['sdf'], out_dict['gradient']
        l = sdf.abs()
            
            
        # eye-balled schedule for loss clamping
        surf_mask = l < 0.1
        l = l[surf_mask]
        if j > int(0.25 * n_steps):
            surf_mask = l < 0.05
            l = l[surf_mask]
        if j > int(0.5 * n_steps):
            surf_mask = l < 0.0075
            l = l[surf_mask]

        loss_dict['surface'] = l.mean()
        
            
        if calculate_normal:
            l_normals = (sdf_grad + sampled_normals).norm(2, dim=-1) #[surf_mask.squeeze(2)]
            sdf_grad_norm = torch.nn.functional.normalize(sdf_grad, dim=-1) 
            l_normals_cosin = 1 + (sdf_grad_norm * sampled_normals).sum(dim=-1, keepdim=True) #[surf_mask]
            loss_dict['normals'] = l_normals.mean()
            loss_dict['normals_cosin'] = l_normals_cosin.mean() 
        else:
            loss_dict['normals'] = 0.0
            loss_dict['normals_cosin'] = 0.0 
        
        if 'normals_eikonal' in lambdas.keys():
            sampled_points_near = sampled_points.clone().detach() + torch.randn((sampled_points.shape[0], sampled_points.shape[1], 3)).float().to(lat_shape.device) * 0.1    
            #previous implementation:            
            #sampled_points_near.requires_grad = True
            #sdf_near = neural3DMM(sampled_points_near, lats_exp, lat_shape)['sdf']
            #_, sdf_grad_near = nabla(neural3DMM, sampled_points_near, lats_exp, lat_shape)
            out_near_dict = neural3DMM({'queries': sampled_points_near}, {'geo': lat_shape.repeat(lats_exp.shape[0], 1, 1), 'exp': lats_exp}, skip_color=True, ignore_deformations=False, return_grad=True)
            sdf_near, sdf_grad_near = out_near_dict['sdf'], out_near_dict['gradient']
            loss_dict['normals_eikonal']  =  ((sdf_grad_near.norm(2, dim=-1) - 1) ** 2).mean()
        else:
            loss_dict['normals_eikonal']  =  0.0
            
        if calculate_anchor:
            anchors = decoder_id.get_anchors(lat_shape)
            query_points = sampled_landmarks
            #previous implementation:   
            #output_expression_decoder = decoder_exp(query_points, lats_exp, lat_shape.repeat(lats_exp.shape[0], 1, 1), anchors.repeat(lats_exp.shape[0], 1, 1))
            in_dict= {'queries': query_points,
                      'cond': {'geo': lat_shape.repeat(lats_exp.shape[0], 1, 1), 'exp': lats_exp},
                      'anchors': anchors.repeat(lats_exp.shape[0], 1, 1),
                      }
            output_expression_decoder = decoder_exp(in_dict)
            
            queries_canonical = query_points + output_expression_decoder['offsets']
            loss_dict['anchors'] = (queries_canonical - anchors[:, anchor_indices, :]).abs().sum(-1).mean()
        else:
            loss_dict['anchors'] = 0.0

        loss_dict['reg_expr'] = torch.norm(torch.norm(lats_exp, dim=-1) ** 2).mean()
            
        # calculate unconditional sds loss inspired by dreamfusion
        if 'sds_expr' in lambdas.keys():
            loss_dict['sds_expr'] = diffusion_expr.forward_sds(lats_exp, t=t, guidance_scale=guidance_scale, grad_scale=grad_scale, skip_unet=skip_unet)
            
        loss = 0
        for k in lambdas.keys():
            loss += loss_dict[k] * lambdas[k]
        loss.backward()
        
        # optimize step
        if opt_iden:
            optimizer_id.step()
        if opt_expre:
            optimizer_exp.step()
        if opt_rigid_scale:
            optimizer_rigid.step() 
         
        if j % 50 ==0:   
            print_str = "Iter: {:5d}".format(j) 
            print_str += " total loss " + " {:02.8f} ".format(loss)
            for k in lambdas.keys():
                print_str += " " + k + " {:02.8f} ".format(loss_dict[k])
            print_str += "\n"
            for k in lambdas.keys():
                print_str +=  "weighted_" + k + " {:02.8f} ".format(loss_dict[k] * lambdas[k])
            print_str += " sds noise timestep:" + " {:03d} ".format(t.squeeze().item())
            print(print_str)
        
        if log_loss:
            wandb.log(loss_dict)

    # np.save('/home/giebenhain/proj1/NPFM/lat_rep_mean.npy', lat_rep.detach().squeeze().cpu().numpy())
    rotation_matrix = []
    rotation_euler = []
    for c in range(len(rotation)):
        if use_quaternion:
            rotation_matrix.append( pytorch3d.transforms.quaternion_to_matrix(rotation[c]) )
        else:
            rotation_matrix.append( pytorch3d.transforms.euler_angles_to_matrix(rotation[c], convention=convention) )
    rotation_matrix = torch.stack(rotation_matrix, axis=0)  
    rotation_euler = rotation.detach()
    return lats_exp, lat_shape, None,  {'rot': rotation_matrix, 'trans': translation, 'scale': scale, 'rot_euler': rotation_euler} 

def inference_incremental_expression(neural3DMM,
                        diffusion_shape, diffusion_expr, 
                        lats_exp, lat_shape, 
                        lambdas, sds_args,  all_points_exp, all_normals_exp, 
                        all_rotation_params, all_translation_params, all_scale_params, all_lms_exp, 
                        anchor_indices=(0, 1, 10, 11, 12, 13, 34, 35), init_lr=0.0002, init_lr2=0.0002, init_lr_rigid=0.000001, n_steps=31, n_steps2=31, 
                        schedule_cfg={}, schedule2_cfg={},  schedule_rigid_cfg={},
                        calculate_normal=False, calculate_anchor=False, opt_rigid_scale=False, opt_iden=True, opt_expre=True,
                        log_loss=False, project="fit_pointclouds_seq", exp_name="nphm_kinect",
                        use_quaternion = False, convention = 'ZYX', flame2scan = False):
    
    if log_loss:
        wandb.init(project=project, config=lambdas, name=exp_name)
    
    t_range = sds_args.get('t_range', [0.02, 0.98])
    guidance_scale = sds_args.get('guidance_scale', 100) 
    grad_scale = sds_args.get('grad_scale', 1) 
    skip_unet = sds_args.get('skip_unet', False)

     # nonrigid parameters
    if opt_iden:
        lat_shape.requires_grad = True
        optimizer_id = optim.Adam(params=[lat_shape], lr=init_lr) 
    
    scale = torch.from_numpy(all_scale_params[0]).to(lat_shape.device)
        
    # per frame parameters
    all_lat_exp = []
    all_rot = []
    all_trans = []
    for i, points_exp in enumerate(all_points_exp):
        assert torch.isnan(points_exp).sum() == 0
    
        lat_exp = lats_exp[i:i+1]
        all_lat_exp.append(lat_exp)

        rotation = (all_rotation_params[i]).to(lat_shape.device) # torch.from_numpy
        translation = (all_translation_params[i]).to(lat_shape.device)
        all_rot.append(rotation)
        all_trans.append(translation)


    all_points_exp_clone = []
    for i, points_exp in enumerate(all_points_exp):
        all_points_exp_clone.append(points_exp.detach().clone().unsqueeze(0))
        assert torch.isnan(points_exp).sum() == 0
    all_normals_exp_clone = []
    for i, normals_exp in enumerate(all_normals_exp):
        all_normals_exp_clone.append(normals_exp.detach().clone().unsqueeze(0))
        assert torch.isnan(normals_exp).sum() == 0

    decoder_id = neural3DMM.id_model #identity_model
    decoder_exp = neural3DMM.ex_model #expression_model
    
    for i in range(1, len(all_points_exp)):
        if i == 0:
            lat_rep_i = all_lat_exp[i]
            lat_rep_i.requires_grad = True
            opt = optim.Adam(params=[lat_rep_i], lr=init_lr)
            n_iter = n_steps
        else:  
            all_lat_exp[i] = all_lat_exp[i-1].detach().clone() 
            all_lat_exp[i].requires_grad = True
            opt = optim.Adam(params=[ all_lat_exp[i] ], lr=init_lr2) 
            n_iter = n_steps2
            
        lat_rep_i = all_lat_exp[i]
        
        if opt_rigid_scale:
            all_rot[i].requires_grad = True
            all_trans[i].requires_grad = True
            optimizer_rigid = optim.Adam(params=[all_rot[i], all_trans[i]], lr=init_lr_rigid)
        
            
        for j in range(n_iter):
            if i == 0:
                if int(j) in schedule_cfg['lr']:
                    for param_group in opt.param_groups:
                        param_group["lr"] /= schedule_cfg['lr'][int(j)]
            else:
                if int(j) in schedule2_cfg['lr']:
                    for param_group in opt.param_groups:
                        param_group["lr"] /= schedule2_cfg['lr'][int(j)]
            if opt_rigid_scale:
                if int(j) in schedule_rigid_cfg['lr']:
                    for param_group in optimizer_rigid.param_groups:
                        param_group["lr"] /= schedule_rigid_cfg['lr'][int(j)]
            
            loss_dict = {}
            opt.zero_grad()
            if opt_rigid_scale:
                optimizer_rigid.zero_grad()
            if opt_iden:
                optimizer_id.zero_grad()

            n_samps = 10000
            subsample_idx = torch.randint(0, all_points_exp_clone[i].shape[1], [n_samps])
            p_exp = all_points_exp_clone[i][:, subsample_idx, :]
            n_exp = all_normals_exp_clone[i][:, subsample_idx, :]
            
            
            
            if flame2scan:
                if use_quaternion:
                    quan_or_euler_to_rot = pytorch3d.transforms.quaternion_to_matrix(all_rot[i])
                else:
                    quan_or_euler_to_rot = pytorch3d.transforms.euler_angles_to_matrix(all_rot[i], convention=convention)
                p_exp_nphm = ( (p_exp.squeeze() - all_trans[i].unsqueeze(0)) @ quan_or_euler_to_rot / scale ).unsqueeze(0)  * 4.0
                n_exp_nphm = (n_exp.squeeze() @ quan_or_euler_to_rot ).unsqueeze(0)
                lms_exp_nphm =  ( (all_lms_exp[i].squeeze() - all_trans[i].unsqueeze(0)) @ quan_or_euler_to_rot / scale ).unsqueeze(0)  * 4.0
            else:
                raise NotImplementedError
            
            
            # previous implementation
            #sdf = neural3DMM(p_exp_nphm, lat_rep_i, lat_shape)['sdf']
            #_, sdf_grad = nabla(neural3DMM, p_exp_nphm, lat_rep_i, lat_shape) 
            out_dict = neural3DMM({'queries': p_exp_nphm}, {'geo': lat_shape, 'exp': lat_rep_i}, skip_color=True, ignore_deformations=False, return_grad=True)
            sdf, sdf_grad = out_dict['sdf'], out_dict['gradient']
            l = sdf.abs()
                
            # eye-balled schedule for loss clamping
            surf_mask = l < 0.1
            l = l[surf_mask]
            if j > int(0.25 * n_steps):
                surf_mask = l < 0.05
                l = l[surf_mask]
            if j > int(0.5 * n_steps):
                surf_mask = l < 0.0075
                l = l[surf_mask]

            loss_dict['surface'] = l.mean()
                
            if calculate_normal:
                l_normals = (sdf_grad + n_exp_nphm).norm(2, dim=-1)
                sdf_grad_norm = torch.nn.functional.normalize(sdf_grad, dim=-1) 
                l_normals_cosin = 1 + ( sdf_grad_norm * n_exp_nphm).sum(dim=-1, keepdim=True)
                loss_dict['normals'] = l_normals.mean()
                loss_dict['normals_cosin'] = l_normals_cosin.mean() 
            else:
                loss_dict['normals'] = 0.0
                loss_dict['normals_cosin'] = 0.0
            
            if 'normals_eikonal' in  lambdas.keys():                
                p_exp_nphm_near = p_exp_nphm.clone().detach() + torch.randn((p_exp_nphm.shape[0], p_exp_nphm.shape[1], 3)).float().to(lat_shape.device) * 0.1                
                #p_exp_nphm_near.requires_grad = True
                #sdf_near = neural3DMM(p_exp_nphm_near, lat_rep_i, lat_shape)['sdf']
                # _, sdf_grad_near = nabla(neural3DMM, p_exp_nphm_near, lat_rep_i, lat_shape) 
                out_near_dict = neural3DMM({'queries': p_exp_nphm_near}, {'geo': lat_shape, 'exp': lat_rep_i}, skip_color=True, ignore_deformations=False, return_grad=True)
                sdf_near, sdf_grad_near = out_near_dict['sdf'], out_near_dict['gradient']
                loss_dict['normals_eikonal']  =  ((sdf_grad_near.norm(2, dim=-1) - 1) ** 2).mean()        
            else:
                loss_dict['normals_eikonal']  = 0.0           
                
            if calculate_anchor:
                anchors = decoder_id.get_anchors(lat_shape)
                query_points = lms_exp_nphm
                # output_expression_decoder = decoder_exp(query_points, lat_rep_i, lat_shape, anchors)
                in_dict= {'queries': query_points,
                      'cond': {'geo': lat_shape, 'exp': lat_rep_i},
                      'anchors': anchors,
                      }
                output_expression_decoder = decoder_exp(in_dict)
            
                queries_canonical = query_points + output_expression_decoder['offsets']
                loss_dict['anchors'] = (queries_canonical - anchors[:, anchor_indices, :]).abs().sum(-1).mean()
            else:
                loss_dict['anchors'] = 0.0

            loss_dict['reg_expr'] = torch.norm(torch.norm(lat_rep_i, dim=-1) ** 2).mean()
            
            min_step =  int(t_range[0] * diffusion_expr.num_timesteps)
            max_step =  int(t_range[1] * diffusion_expr.num_timesteps) + 1
            t = torch.randint(min_step, max_step, (lat_shape.size(0),), device=lat_shape.device).long()
            if 'sds_expr' in lambdas.keys():    
                # calculate unconditional sds loss inspired by dreamfusion
                loss_dict['sds_expr'] = diffusion_expr.forward_sds(lat_rep_i, t=t,  guidance_scale=guidance_scale, grad_scale=grad_scale, skip_unet=skip_unet)
                glob_cond = torch.cat([ lat_shape, lat_rep_i ], dim=-1)
            else:
                loss_dict['sds_expr'] = 0.0
                
            # regularize rigid transform params
            if opt_rigid_scale: 
                if i == 0: 
                    loss_dict['reg_rot'] = 0.0
                    loss_dict['reg_trans'] = 0.0
                else:
                    loss_dict['reg_rot'] = torch.norm( all_rot[i] - all_rot[i-1] )
                    loss_dict['reg_trans'] = torch.norm( all_trans[i] - all_trans[i-1] )
                loss_dict['reg_scale'] = 0.0
            else:
                loss_dict['reg_rot'] = 0.0
                loss_dict['reg_trans'] = 0.0
                loss_dict['reg_scale'] = 0.0
                
            # include expression smooth loss
            if i > 0:
                loss_dict['reg_expre_smo'] = (torch.norm(lat_rep_i - all_lat_exp[i-1].detach().clone(), dim=-1) ** 2).mean()      
            else:
                loss_dict['reg_expre_smo'] = 0.0 
                
            loss = 0
            for k in lambdas.keys():
                loss += loss_dict[k] * lambdas[k]
            loss.backward()
            
            # optimize 
            opt.step()
            
            if opt_rigid_scale:
                optimizer_rigid.step()
                
            if opt_iden:
                optimizer_id.step()
            
            if log_loss:
                wandb.log(loss_dict)

            if j % 5 ==0 or j == n_iter-1:
                print_str = "Num obs: {:5d}".format(i) + " " + "Epoch: {:5d}".format(j)
                for k in lambdas.keys():
                    print_str += " " + k + " {:02.8f} ".format(loss_dict[k])
                print_str += "\n"
                for k in lambdas.keys():
                    print_str += " weighted_" + k + " {:02.8f} ".format(loss_dict[k] * lambdas[k])
                print_str += " sds noise timestep:" + " {:03d} ".format(t.squeeze().item())
                print(print_str)

        print("update lat_rep_expre of obs {:5d}".format(i) )
        ##all_lat_exp[i] = lat_rep_i.detach().clone()

    rotation_matrix = []
    rotation_euler = []
    for c in range(len(all_rot)):
        if use_quaternion:
            rotation_matrix.append( pytorch3d.transforms.quaternion_to_matrix(all_rot[c]) )
        else:
            rotation_matrix.append( pytorch3d.transforms.euler_angles_to_matrix(all_rot[c], convention=convention) )
    rotation_matrix = torch.stack(rotation_matrix, axis=0)  
    rotation_euler = torch.stack(all_rot, dim=0)
    translation = torch.stack( all_trans, dim=0)     
    lats_exp = torch.concat(all_lat_exp, dim=0)  
    return lats_exp, lat_shape, None,  {'rot': rotation_matrix, 'trans': translation, 'scale': scale, 'rot_euler': rotation_euler} 


def inference_joint_chunk_fitting(neural3DMM,
                        diffusion_shape, diffusion_expr, 
                        lats_exp, lat_shape, 
                        lambdas, sds_args,  all_points_exp, all_normals_exp, 
                        all_rotation_params, all_translation_params, all_scale_params, all_lms_exp, 
                        anchor_indices=(0, 1, 10, 11, 12, 13, 34, 35), init_lr=0.0002, init_lr_rigid=0.000001, n_steps=31, 
                        schedule_cfg={}, schedule_rigid_cfg={},
                        calculate_normal=False, calculate_anchor=False, opt_rigid=True, opt_scale=True, opt_iden=True, opt_expre=True,
                        log_loss=False, project="fit_pointclouds_seq", exp_name="nphm_kinect",
                        use_quaternion = False, convention = 'ZYX', flame2scan = False):
    
    if log_loss:
        wandb.init(project=project, config=lambdas, name=exp_name)
        #wandb.watch(decoder_exp, log_freq=100)
        
    t_range = sds_args.get('t_range', [0.02, 0.98])
    guidance_scale = sds_args.get('guidance_scale', 100) 
    grad_scale = sds_args.get('grad_scale', 1) 
    skip_unet = sds_args.get('skip_unet', False)

    decoder_id = neural3DMM.id_model #identity_model
    decoder_exp = neural3DMM.ex_model #expression_model
    
    # nonrigid parameters
    if opt_iden:
        lat_shape.requires_grad = True
        optimizer_id = optim.Adam(params=[lat_shape], lr=init_lr) 
    
    scale = torch.from_numpy(all_scale_params[0]).to(lat_shape.device)
    if opt_scale:
        scale.requires_grad = True
        optimizer_scale = torch.optim.Adam(params=[scale], lr=init_lr_rigid)
        
    # per frame parameters
    all_lat_exp = []
    all_opt_exp = []
    all_rot = []
    all_trans = []
    all_opt_rigid = []
    for i, points_exp in enumerate(all_points_exp):
        assert torch.isnan(points_exp).sum() == 0

        if opt_expre:
            lats_exp[i:i+1].requires_grad = True
            optimizer_expr = optim.Adam(params=[lats_exp[i:i+1]], lr=init_lr)  # TODO what LR is best?
            all_opt_exp.append(optimizer_expr)

        if opt_rigid:
            all_rotation_params[i].requires_grad = True
            all_translation_params[i].requires_grad = True
            optimizer_rigid = optim.Adam(params=[all_rotation_params[i], all_translation_params[i]], lr=init_lr_rigid) #0.000001 
            all_opt_rigid.append(optimizer_rigid)


    all_points_exp_clone = []
    for i, points_exp in enumerate(all_points_exp):
        all_points_exp_clone.append(points_exp.detach().clone().unsqueeze(0))
        assert torch.isnan(points_exp).sum() == 0
    all_normals_exp_clone = []
    for i, normals_exp in enumerate(all_normals_exp):
        all_normals_exp_clone.append(normals_exp.detach().clone().unsqueeze(0))
        assert torch.isnan(normals_exp).sum() == 0


    decoder_id = neural3DMM.id_model #identity_model
    decoder_exp = neural3DMM.ex_model #expression_model

    all_frames = list(range(len(all_points_exp)))
    chunk_size = 2
    #num_steps = [0 for _ in range(len(all_frames))]

    for j in range(n_steps):
        if int(j) in schedule_cfg['lr']:
            if opt_iden:
                for param_group in optimizer_id.param_groups:
                    param_group["lr"] /= schedule_cfg['lr'][int(j)]
                print('reduced LR for id')
            if opt_expre:
                for opt in all_opt_exp:
                    for param_group in opt.param_groups:
                        param_group["lr"] /= schedule_cfg['lr'][int(j)]
                print('reduced LR for expr')


        if int(j) in schedule_rigid_cfg['lr']:
            if opt_scale:
                for param_group in optimizer_scale.param_groups:
                    param_group["lr"] /= schedule_rigid_cfg['lr'][int(j)]
                print('reduced LR for scale')

            if opt_rigid:
                for opt_rigid in all_opt_rigid:
                    for param_group in opt_rigid.param_groups:
                        param_group["lr"] /= schedule_rigid_cfg['lr'][int(j)]
                print('reduced LR for rigid')


        sum_loss_dict = {k: 0.0 for k in lambdas}
        sum_loss_dict.update({'loss': 0.0})
        print(sum_loss_dict)
        
        perm = np.random.permutation(len(all_frames) - chunk_size+1)
        for c_ in range(perm.shape[0]):
            cs = list(range(perm[c_], perm[c_]+chunk_size))
            assert len(cs) == chunk_size

            if opt_iden:
                optimizer_id.zero_grad()

            if opt_expre:
                for opt in all_opt_exp:
                    opt.zero_grad()

            if opt_scale:
                optimizer_scale.zero_grad()

            if opt_rigid:
                for opt_rigid in all_opt_rigid:
                    opt_rigid.zero_grad()
                
            loss_dict = {'smooth_expr': 0, 'smooth_rot': 0, 'smooth_trans': 0, 
                         'surface': 0,  'normals': 0, 'normals_cosin': 0, 'anchors': 0,
                         # specific to global params
                         'symm_dist': 0, 'reg_loc': 0, 'reg_glob': 0,  'reg_extra': 0, 'reg_expr': 0,  
                         'sds_shape': 0, 'sds_expr':0, 
                         }

            # identity code regularization
            loss_dict['reg_glob'] = (torch.norm(lat_shape[..., :decoder_id.lat_dim_glob], dim=-1) ** 2).mean() * chunk_size
            loss_dict['reg_loc'] = ((torch.norm(lat_shape[..., decoder_id.lat_dim_glob:], dim=-1) ** 2)).mean() * chunk_size

            loc_lats_symm = lat_shape[:, :,
                            decoder_id.lat_dim_glob:decoder_id.lat_dim_glob + 2 * decoder_id.num_symm_pairs * decoder_id.lat_dim_loc].view(
                lat_shape.shape[0], decoder_id.num_symm_pairs * 2, decoder_id.lat_dim_loc)

            symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()

            symm_idx = [ (lables.index('eye_inner_left'), lables.index('eye_inner_right')),
                        (lables.index('eye_outer_left'), lables.index('eye_outer_right')),
                        (lables.index('mouth_corner_l'), lables.index('mouth_corner_r'))]
            symm_dist = 0
            for symm_pair in symm_idx:
                symm_dist += torch.norm(lat_shape[..., decoder_id.lat_dim_glob + symm_pair[0]*decoder_id.lat_dim_loc:decoder_id.lat_dim_glob + (symm_pair[0]+1)*decoder_id.lat_dim_loc] -
                                        lat_shape[..., decoder_id.lat_dim_glob + symm_pair[1]*decoder_id.lat_dim_loc:decoder_id.lat_dim_glob + (symm_pair[1]+1)*decoder_id.lat_dim_loc],
                                                    dim=-1).square().mean() * chunk_size
            loss_dict['symm_dist'] = symm_dist * chunk_size

            extra_idx = [lables.index('lower_lip'), lables.index('mouth_corner_l'),
                        lables.index('mouth_corner_r')]  # + [10, 11, 12, 13]
            
            for idx in extra_idx:
                start, end = decoder_id.lat_dim_glob + idx * decoder_id.lat_dim_loc, decoder_id.lat_dim_glob + (idx + 1) * decoder_id.lat_dim_loc
                loss_dict['reg_extra'] += torch.norm(lat_shape[..., start:end], dim=-1).square().mean() * chunk_size
            
            
            min_step =  int(t_range[0] * diffusion_shape.num_timesteps)
            max_step =  int(t_range[1] * diffusion_shape.num_timesteps) + 1
            t = torch.randint(min_step, max_step, (lat_shape.size(0),), device=lat_shape.device).long()
            if 'sds_shape' in lambdas.keys():          
                # calculate unconditional sds loss inspired by dreamfusion
                loss_dict['sds_shape'] += diffusion_shape.forward_sds(lat_shape, t=t,  guidance_scale=guidance_scale, grad_scale=grad_scale, skip_unet=skip_unet) * chunk_size
            else:
                loss_dict['sds_shape'] += 0.0

            #for c in cs:
            for cc, c in enumerate(cs):
                n_samps = 5000 #10000
                subsample_idx = torch.randint(0, all_points_exp_clone[c].shape[1], [n_samps])
                p_exp = all_points_exp_clone[c][:, subsample_idx, :]
                n_exp = all_normals_exp_clone[c][:, subsample_idx, :]
                
                
                if flame2scan:
                    if use_quaternion:
                        quan_or_euler_to_rot = pytorch3d.transforms.quaternion_to_matrix(all_rotation_params[c])
                    else:
                        quan_or_euler_to_rot = pytorch3d.transforms.euler_angles_to_matrix(all_rotation_params[c], convention=convention)
                    p_exp_nphm = ( (p_exp.squeeze() - all_translation_params[c].unsqueeze(0)) @ quan_or_euler_to_rot / scale ).unsqueeze(0)  * 4.0
                    n_exp_nphm = (n_exp.squeeze() @ quan_or_euler_to_rot ).unsqueeze(0)
                    lms_exp_nphm =  ( (all_lms_exp[c].squeeze() - all_translation_params[c].unsqueeze(0)) @ quan_or_euler_to_rot / scale ).unsqueeze(0)  * 4.0
                else:
                    raise NotImplementedError

                #sdf = neural3DMM(p_exp_nphm, lats_exp[c:c+1], lat_shape)['sdf']
                #_, sdf_grad = nabla(neural3DMM, p_exp_nphm, lats_exp[c:c+1], lat_shape)
                out_dict = neural3DMM({'queries': p_exp_nphm}, {'geo': lat_shape, 'exp': lats_exp[c:c+1]}, skip_color=True, ignore_deformations=False, return_grad=True)
                sdf, sdf_grad = out_dict['sdf'], out_dict['gradient']
                l = sdf.abs()

                # eye-balled schedule for loss clamping
                surf_mask = l < 0.0075
                l = l[surf_mask]

                loss_dict['surface'] += l.mean()
                    
                if calculate_normal:
                    l_normals = (sdf_grad + n_exp_nphm).norm(2, dim=-1)#[surf_mask]
                    sdf_grad_norm = torch.nn.functional.normalize(sdf_grad, dim=-1) 
                    l_normals_cosin = 1 + ( sdf_grad_norm * n_exp_nphm).sum(dim=-1, keepdim=True)#[surf_mask]
                    loss_dict['normals'] = l_normals.mean()
                    loss_dict['normals_cosin'] = l_normals_cosin.mean() 
                else:
                    loss_dict['normals'] += 0.0
                    loss_dict['normals_cosin'] += 0
                
                if calculate_anchor:
                    anchors = decoder_id.get_anchors(lat_shape)
                    query_points = lms_exp_nphm
                    #output_expression_decoder = decoder_exp(query_points, lats_exp[c:c+1], lat_shape, anchors) 
                    in_dict= {'queries': query_points,
                        'cond': { 'geo': lat_shape, 'exp': lats_exp[c:c+1] },
                        'anchors': anchors,
                        }
                    output_expression_decoder = decoder_exp(in_dict)
                        
                    queries_canonical = query_points + output_expression_decoder['offsets']
                    loss_dict['anchors'] = (queries_canonical - anchors[:, anchor_indices, :]).abs().sum(-1).mean()
                else:
                    loss_dict['anchors'] += 0.0

                loss_dict['reg_expr'] += torch.norm(torch.norm(lats_exp[c:c+1], dim=-1) ** 2).mean()
                    
                if 'sds_expr' in lambdas.keys():
                    # calculate unconditional sds loss inspired by dreamfusion
                    loss_dict['sds_expr'] += diffusion_expr.forward_sds(lats_exp[c:c+1],  t=t, guidance_scale=guidance_scale, grad_scale=grad_scale, skip_unet=skip_unet)
                    glob_cond = torch.cat([ lat_shape, lats_exp[c:c+1] ], dim=-1)
                else:
                    loss_dict['sds_expr']  += 0.0
                    
                if c + 1 < len(cs):
                    loss_dict['smooth_expr'] += ( lats_exp[c:c+1] - lats_exp[c+1:c+2] ).norm(dim=-1).square().mean() 
                    loss_dict['smooth_rot'] += (all_rotation_params[c] - all_rotation_params[c+1]).norm(dim=-1).mean()  
                    loss_dict['smooth_trans'] += (all_translation_params[c] - all_translation_params[c+1]).norm(dim=-1).mean() 
                if c > 0:
                    loss_dict['smooth_expr'] += ( lats_exp[c:c+1] - lats_exp[c-1:c] ).norm(dim=-1).square().mean()  
                    loss_dict['smooth_rot'] += (all_rotation_params[c] - all_rotation_params[c-1]).norm(dim=-1).mean() 
                    loss_dict['smooth_trans'] += (all_translation_params[c] - all_translation_params[c-1]).norm(dim=-1).mean()
                

            loss = 0
            for k in lambdas.keys():
                loss += loss_dict[k]/chunk_size * lambdas[k]
            loss.backward()
            
            # optimize step
            if opt_iden:
                optimizer_id.step()
            if opt_scale:
                optimizer_scale.step()
            #for c in cs:
            for cc, c in enumerate(cs):
                if opt_expre:
                    all_opt_exp[c].step()
                if opt_rigid:
                    all_opt_rigid[c].step()
                
            # calculate the sum
            #for k in loss_dict:
            for k in lambdas.keys():
                sum_loss_dict[k] += loss_dict[k]/chunk_size * lambdas[k]
            sum_loss_dict['loss'] += loss
            
            if c_ % 100 == 0: #20 60 100
                print_str = "Epoch: {:5d}".format(j) + " Chunk: {:5d}".format(c_) 
                for k in lambdas.keys():
                    print_str += " " + k + " {:02.8f} ".format(loss_dict[k]/chunk_size)
                print_str += "\n"
                for k in lambdas.keys():
                    print_str += " weighted_" + k + " {:02.8f} ".format(loss_dict[k]/chunk_size * lambdas[k])
                print_str += " sds noise timestep:" + " {:03d} ".format(t.squeeze().item())
                print(print_str)
            
        # and then avg
        for k in sum_loss_dict:
            sum_loss_dict[k] /= perm.shape[0] 
        
        print_str = "Epoch: {:5d}".format(j) 
        for k in lambdas.keys():
            print_str += "AVG weighted" + k + " {:02.8f} ".format(sum_loss_dict[k])
        print(print_str)

        if log_loss:
            wandb.log(sum_loss_dict)

    rotation_matrix = []
    rotation_euler = []
    for c in range(len(all_rotation_params)):
        if use_quaternion:
            rotation_matrix.append( pytorch3d.transforms.quaternion_to_matrix(all_rotation_params[c]) )
        else:
            rotation_matrix.append( pytorch3d.transforms.euler_angles_to_matrix(all_rotation_params[c], convention=convention) )
    rotation_matrix = torch.stack(rotation_matrix, axis=0)  
    rotation_euler = torch.stack(all_rotation_params, dim=0)
    translation = torch.stack( all_translation_params, dim=0)     
    
    return lats_exp, lat_shape, None, {'rot': rotation_matrix, 'trans': translation, 'scale': scale, 'rot_euler': rotation_euler} 