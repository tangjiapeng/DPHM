import yaml
import os
import wandb
import trimesh
import datetime
import argparse
import numpy as np
import pickle
from glob import glob
import torch.multiprocessing
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import os.path as osp
import math

from dphm_tum.flame.flame_model import FLAME
from dphm_tum.utils.flame_utils import matrix_to_euler_angles, euler_angles_to_matrix
from dphm_tum.utils.io import export_pointcloud_o3d
from dphm_tum.utils.landmark import IBU68_index_into_WFLW
from dphm_tum.utils.pyvista_render import render_snapshot, render_from_camera, blend_color_and_recon, calcaute_nearsest_distance

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from dataset_creation.flame_deformation import DeformationFLAME
from dataset_creation.alignment import align_flame_to_multi_view_stereo, align_flame_to_scans_icp
from dataset_creation.coordinate_transform import invert_similarity_transformation, rigid_transform

torch.multiprocessing.set_sharing_strategy('file_system')  # a fix for the "OSError: too many files" exception
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def optimize_flame_parameters(args):
    with open('./configs/flame/flame.yaml', "r") as yamlfile:
        conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
        config = OmegaConf.create(conf)
    
    data_path = config.kinect_data_path
    subjects = sorted(os.listdir(data_path)) 
    outdir = config.kinect_flame_fit_dir
    pc_crop_foldername =  config.kinect.pc_folder #"points_new"
    nms_crop_foldername = config.kinect.nms_folder #"normals_new"
    lms_pip_foldername = config.kinect.pip_folder #"lms_3d_pip"
    lms_pip_crop_foldername = config.kinect.mp_folder #"lms_3d_pip_cropped3",
    subsample_npoints = config.kinect.subsample_npoints #100000
    down_ratio = config.kinect.down_ratio # 1
    
    _use_normals = config.kinect.use_normals # True
    _use_landmark = config.kinect.use_landmark # True
    _use_jaw = config.kinect.use_jaw  # false
    _mask_forehead = config.kinect.mask_forehead  # true
    _from_flame_to_scan = config.kinect.from_flame_to_scan # true
    _use_quanternion =  config.kinect.use_quanternion # false
    _euler_convention = config.kinect.euler_convention # 'XYZ'
    _selected_sequences = config.kinect.subjects
    
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # TODO: Create the models and set device
    _flame = FLAME(config)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flame = _flame.to(_device)
    
    for subj in _selected_sequences[args.start_idx: args.end_idx]:        
        number_of_frames = len([ fns for fns in os.listdir(osp.join(data_path, subj, pc_crop_foldername)) if fns.endswith('.npy') ])
        num_frames = number_of_frames
        frame_inds = list(range(0, number_of_frames, down_ratio))
        frame_inds = [ frame for frame in frame_inds if osp.exists( osp.join(data_path, subj, '{:s}/{:05d}.npy'.format(pc_crop_foldername, frame)) ) ]
        
        print(outdir, subj)
        mapping_dir = os.path.join(outdir, f"{subj}")
        print(mapping_dir)
        flame_registration_param_filename = 'flame_parameters'
        if _from_flame_to_scan:
            flame_registration_param_filename += '_flame2mvs'
        else:
            flame_registration_param_filename += '_mvs2flame'
        if _use_quanternion:
            flame_registration_param_filename += '_quan'
        
        if os.path.exists(f"{mapping_dir}/{flame_registration_param_filename}.npz"):
            print(f"Skipping {mapping_dir} as it is already processed")
            continue
        
        all_points3d = []
        all_normals3d = []
        all_landmarks3d = []

        for frame in frame_inds:
            points = np.load(osp.join(data_path, subj, '{:s}/{:05d}.npy'.format(pc_crop_foldername, frame)))
            normals = np.load(osp.join(data_path, subj, '{:s}/{:05d}.npy'.format(nms_crop_foldername, frame)))
            rnd_ind = torch.randint(0, points.shape[0], size=[subsample_npoints])
            points_subsample = points[rnd_ind, :]
            normals_subsample = normals[rnd_ind, :]
            landmarks3d = np.load(osp.join(data_path, subj, '{:s}/{:05d}.npy'.format(lms_pip_foldername, frame)))[IBU68_index_into_WFLW]
            all_points3d.append(points_subsample)
            all_normals3d.append(normals_subsample)
            all_landmarks3d.append(landmarks3d)

        # Perform flame fitting
        # Create zero latent codes
        _flame_shape_params = torch.zeros(1, config.z_shape_flame).float()
        _flame_shape_params = _flame_shape_params.to(_device)

        # Per frame parameters
        _flame_expressions = []
        _flame_rotations = []
        _flame_translations = []
        _flame_jaw_poses = []
        _flame_scales = []

        # Flame template mesh
        flame_template_mesh = trimesh.load(config.flame_average_mesh_path)

        print("Perform Rigid Alignment")
        # Rigid Alignment
        for idx in range(len(all_landmarks3d)):
            print("run initial rigid alignments for {:d} of {:d}".format(idx, len(all_landmarks3d)))
            _flame_expressions.append(torch.zeros(1, config.z_expression_flame).float().to(_device))
            if _from_flame_to_scan:
                # use some of jawline
                s, R, t = align_flame_to_multi_view_stereo(flame_template_mesh, all_landmarks3d[idx], kinect=True)
            else:
                s, R, t = align_flame_to_multi_view_stereo(flame_template_mesh, all_landmarks3d[idx], reverse=True, kinect=True)
            _flame_scales.append(s)
            if _use_quanternion:
                quanternion = matrix_to_quaternion(torch.from_numpy(R))
                _flame_rotations.append(quanternion.float().to(_device).unsqueeze(0))
            else:
                euler_rotations = matrix_to_euler_angles(torch.from_numpy(R), _euler_convention)
                _flame_rotations.append(euler_rotations.float().to(_device).unsqueeze(0))
            _flame_translations.append(torch.from_numpy(t).float().to(_device).unsqueeze(0))
            _flame_jaw_poses.append(torch.zeros(1, 3).float().to(_device))

        flame_parameters = {}
        flame_parameters['shape'] = _flame_shape_params.detach().requires_grad_()
        flame_parameters['expression'] = torch.cat(_flame_expressions, dim=0).detach().requires_grad_()
        flame_parameters['rotation'] = torch.cat(_flame_rotations, dim=0).detach().requires_grad_()
        flame_parameters['translation'] = torch.cat(_flame_translations, dim=0).detach().requires_grad_()
        flame_parameters['jaw_pose'] = torch.cat(_flame_jaw_poses, dim=0).detach().requires_grad_()
        _mean_flame_scale = np.mean(np.array(_flame_scales))
        flame_parameters['scale'] = _mean_flame_scale * torch.ones([1, 1]).float().to(_device).detach()
        flame_parameters['scale'].requires_grad = True
        
        num_frames = flame_parameters['expression'].shape[0]
        # Optimize Flame parameters
        deform_flame = DeformationFLAME(config, all_point3d=all_points3d, all_normals3d=all_normals3d, all_landmarks3d=all_landmarks3d,
                                        flame_parameters=flame_parameters, _device=_device, 
                                        use_normals=_use_normals,  use_landmark=_use_landmark, use_jaw=_use_jaw, mask_forehead=_mask_forehead, use_quanternion=_use_quanternion, euler_convention=_euler_convention, from_flame_to_scan=_from_flame_to_scan)
        optimizer = torch.optim.Adam([flame_parameters['shape'], flame_parameters['expression'], flame_parameters['jaw_pose'],  flame_parameters['translation'], flame_parameters['scale'] ], lr=0.01) #for better temporal conherence ? # 0.01 for students and changluo, weicao
        # Optimize rigid transformation parameters
        optimizer_rigid = torch.optim.Adam([ flame_parameters['rotation'] ], lr=0.01)
        
        alternative = True
        n_iterations = 1000 
        n_step = 400
        print("Optimizing for landmarks and geometric constraints...")
        experiment = f"{'kinect'}_{subj}"
        if args.log_loss:
            experiment = f"{datetime.datetime.now().strftime('%d%m%H%M')}_{'identity'}_{subj}_{'kinect'}"
            wandb.init(project='FlameFitting', name=experiment)

        for step_idx in tqdm(range(n_iterations)):
            if alternative:
                if step_idx % 2 == 1:
                    optimizer.zero_grad()
                if step_idx % 2 == 0:
                    optimizer_rigid.zero_grad()
            else:
                optimizer.zero_grad()
                optimizer_rigid.zero_grad()
            loss_dict = deform_flame(flame_parameters, step_idx)


            loss_dict["total_loss"].backward()
            if alternative:
                if step_idx % 2 == 1:
                    optimizer.step()
                if step_idx % 2 == 0:
                    optimizer_rigid.step()
            else:
                optimizer.step()
                optimizer_rigid.step()

            # Lower learning rate after 2500 iterations
            if (step_idx+1) % n_step == 0:
                for g in optimizer.param_groups:
                    g['lr'] /= 10
                for g in optimizer_rigid.param_groups:
                    g['lr'] /= 10
                

            if step_idx % 20 == 0 and args.log_loss:
                wandb.log({'step': step_idx, f'total_loss': loss_dict["total_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'landmark2d_loss': loss_dict["landmark2d_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'landmark_loss': loss_dict["landmark_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'geometric_loss': loss_dict["geometric_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'shape_params_loss': loss_dict["shape_params_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'expression_params_loss': loss_dict["expression_params_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'pose_params_loss': loss_dict["pose_params_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'rigid_transform_loss': loss_dict["rigid_transform_loss"]}, step=step_idx)
                wandb.log({'step': step_idx, f'expression_smoothness': loss_dict["expression_smoothness"]}, step=step_idx)
                wandb.log({'step': step_idx, f'pose_smoothness': loss_dict["pose_smoothness"]}, step=step_idx)
                wandb.log({'step': step_idx, f'rigid_smoothness': loss_dict["rigid_smoothness"]}, step=step_idx)
                
            print_str = "Iter: {:5d}".format(step_idx)
            for k in loss_dict.keys():
                print_str += " " + k + " {:02.8f} ".format(loss_dict[k].item())
            print(print_str)
    

        #meshes_flame_space, meshes_mvs_space = deform_flame.visualize(flame_parameters, visualize=False)
        meshes_flame_space, landmark_flame_space, meshes_mvs_space, landmark_mvs_space, points_obs, normals_obs, landmark_obs, points_obs_flame_space, normals_obs_flame_space, landmark_obs_flame_space = deform_flame.visualize(flame_parameters, export_dir_base=mapping_dir, visualize=True)
        if _use_quanternion:
            if _from_flame_to_scan:
                global_rotation = np.stack([quaternion_to_matrix(flame_parameters['rotation'][i]).squeeze().detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) # num_frames!!!!!
                param_dict = {k: np.stack([v[i].detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) for (k, v) in flame_parameters.items() if k != 'shape' and k != 'scale'} # num_frames!!!!!
                param_dict.update({'shape': flame_parameters['shape'].detach().cpu().numpy(),
                                'scale': flame_parameters['scale'].detach().cpu().numpy()})
                np.savez(mapping_dir + f'/flame_parameters_flame2mvs_quan_raw.npz', shape=param_dict['shape'], expression=param_dict['expression'],
                    rotation=param_dict['rotation'], rotation_matrices=global_rotation, translation=param_dict['translation'],
                    jaw=param_dict['jaw_pose'], frames=np.array(num_frames), scale=param_dict['scale'])
                
                ###
                global_rotation = np.stack([quaternion_to_matrix(flame_parameters['rotation'][i].T).squeeze().detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) # num_frames!!!!!
                translations = []
                for i in range(len(frame_inds)):
                    s, R, t = flame_parameters['scale'], quaternion_to_matrix(flame_parameters['rotation'][i]), flame_parameters['translation'][i]
                    translations.append( -( 1 /s ) * t @ R  )
                translations = torch.concat(translations, dim=0)
                param_dict = {k: np.stack([v[i].detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) for (k, v) in flame_parameters.items() if k != 'shape' and k != 'scale'} # num_frames!!!!!
                param_dict.update({'shape': flame_parameters['shape'].detach().cpu().numpy(),
                                'scale': 1.0 / flame_parameters['scale'].detach().cpu().numpy(),
                                'translation': translations.detach().cpu().numpy() })
            else:
                global_rotation = np.stack([quaternion_to_matrix(flame_parameters['rotation'][i]).squeeze().detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) # num_frames!!!!!
                param_dict = {k: np.stack([v[i].detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) for (k, v) in flame_parameters.items() if k != 'shape' and k != 'scale'} # num_frames!!!!!
                param_dict.update({'shape': flame_parameters['shape'].detach().cpu().numpy(),
                                'scale': flame_parameters['scale'].detach().cpu().numpy()})
        else:
            if _from_flame_to_scan:
                global_rotation = np.stack([euler_angles_to_matrix(flame_parameters['rotation'][i], _euler_convention).squeeze().detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) # num_frames!!!!!
                param_dict = {k: np.stack([v[i].detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) for (k, v) in flame_parameters.items() if k != 'shape' and k != 'scale'} # num_frames!!!!!
                param_dict.update({'shape': flame_parameters['shape'].detach().cpu().numpy(),
                                'scale': flame_parameters['scale'].detach().cpu().numpy()})
                np.savez(mapping_dir + f'/flame_parameters_flame2mvs_raw.npz', shape=param_dict['shape'], expression=param_dict['expression'],
                    rotation=param_dict['rotation'], rotation_matrices=global_rotation, translation=param_dict['translation'],
                    jaw=param_dict['jaw_pose'], frames=np.array(num_frames), scale=param_dict['scale'])
                
                ###
                global_rotation = np.stack([euler_angles_to_matrix(flame_parameters['rotation'][i].T, _euler_convention).squeeze().detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) # num_frames!!!!!
                translations = []
                for i in range(len(frame_inds)):
                    s, R, t = flame_parameters['scale'], euler_angles_to_matrix(flame_parameters['rotation'][i], _euler_convention), flame_parameters['translation'][i]
                    translations.append( -( 1 /s ) * t @ R  )
                translations = torch.concat(translations, dim=0)
                param_dict = {k: np.stack([v[i].detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) for (k, v) in flame_parameters.items() if k != 'shape' and k != 'scale'} # num_frames!!!!!
                param_dict.update({'shape': flame_parameters['shape'].detach().cpu().numpy(),
                                'scale': 1.0 / flame_parameters['scale'].detach().cpu().numpy(),
                                'translation': translations.detach().cpu().numpy() })
            else:
                global_rotation = np.stack([euler_angles_to_matrix(flame_parameters['rotation'][i], _euler_convention).squeeze().detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) # num_frames!!!!!
                param_dict = {k: np.stack([v[i].detach().cpu().numpy() for i in range(len(frame_inds))], axis=0) for (k, v) in flame_parameters.items() if k != 'shape' and k != 'scale'} # num_frames!!!!!
                param_dict.update({'shape': flame_parameters['shape'].detach().cpu().numpy(),
                                'scale': flame_parameters['scale'].detach().cpu().numpy()})

        np.savez(mapping_dir + f'/{flame_registration_param_filename}.npz', shape=param_dict['shape'], expression=param_dict['expression'],
                rotation=param_dict['rotation'], rotation_matrices=global_rotation, translation=param_dict['translation'],
                jaw=param_dict['jaw_pose'], frames=np.array(num_frames), scale=param_dict['scale'])
        
        flame_mesh_dir = os.path.join(mapping_dir, 'meshes_flame')
        os.makedirs(flame_mesh_dir, exist_ok=True)
        mvs_flame_space_dir = os.path.join(mapping_dir, 'scans_flame')
        os.makedirs(mvs_flame_space_dir, exist_ok=True)
        flame_mvs_space_mesh_dir = os.path.join(mapping_dir, 'meshes_kinect')
        os.makedirs(flame_mvs_space_mesh_dir, exist_ok=True)
        mvs_space_mesh_dir = os.path.join(mapping_dir, 'scans_kinect')
        os.makedirs(mvs_space_mesh_dir, exist_ok=True)

        # Save the meshes after training
        for i, frame in enumerate(frame_inds):
            meshes_flame_space[frame].export(os.path.join(flame_mesh_dir, '{:05d}.ply'.format(frame)))
            _ = render_snapshot(os.path.join(flame_mesh_dir, '{:05d}.jpg'.format(frame)), mesh=meshes_flame_space[frame], vis_mesh=True, nphm_coord=True, black=False) # anchors=landmark_flame_space[frame],   vis_anchors=True,
            
            rand_idx = np.random.randint(0, points_obs_flame_space[frame].shape[0], size=10000) 
            export_pointcloud_o3d( os.path.join(mvs_flame_space_dir, '{:05d}_1w.ply'.format(frame)), points_obs_flame_space[frame][rand_idx], normals=normals_obs_flame_space[frame][rand_idx] ) 
            _ = render_snapshot( os.path.join(mvs_flame_space_dir, '{:05d}.jpg'.format(frame)),  points=points_obs_flame_space[frame],  normals=normals_obs_flame_space[frame], vis_points=True, nphm_coord=True, black=False) #anchors=landmark_obs[frame], vis_anchors=True,
            
            meshes_mvs_space[frame].export(os.path.join(flame_mvs_space_mesh_dir, '{:05d}.ply'.format(frame)))
            mesh_verts_dist = calcaute_nearsest_distance(np.array(meshes_mvs_space[frame].vertices), points_obs[frame], delta_dist=0.006)
            _ = render_from_camera(os.path.join(flame_mvs_space_mesh_dir, '{:05d}.jpg'.format(frame)), mesh=meshes_mvs_space[frame], anchors=landmark_obs[frame], mesh_errors=mesh_verts_dist, vis_mesh=True, vis_anchors=True, black=False)
            
            rand_idx = np.random.randint(0, points_obs[frame].shape[0], size=10000)  
            export_pointcloud_o3d( os.path.join(mvs_space_mesh_dir, '{:05d}_1w.ply'.format(frame)), points_obs[frame][rand_idx], normals=normals_obs[frame][rand_idx] ) 
            _ = render_from_camera( os.path.join(mvs_space_mesh_dir, '{:05d}.jpg'.format(frame)),  points=points_obs[frame], anchors=landmark_mvs_space[frame], normals=normals_obs[frame], vis_points=True, vis_anchors=True, black=False)
            print(frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', '-s', type=int, default=0, help='config file with parameters of the model')
    parser.add_argument('--end_idx', '-e',  type=int, default=1000, help='config file with parameters of the model')
    parser.add_argument('--log_loss', '-log', action='store_true', help='config file with parameters of the model')
    args = parser.parse_args()
    
    optimize_flame_parameters(args)