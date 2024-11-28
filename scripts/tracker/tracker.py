from PIL import Image
import numpy as np
import argparse
import json, yaml
import os
import os.path as osp
import torch
import pyvista as pv
import trimesh
from glob import glob
import cv2 as cv
import pickle
from omegaconf import OmegaConf
from scipy.spatial import KDTree, cKDTree
import sys
from pytorch3d.transforms import  matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.ops import knn_points, knn_gather

from dphm_tum import env_paths
from dphm_tum.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from dphm_tum.models.reconstruction import get_logits, get_canonical_vertices
from dphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint
from dphm_tum.models.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Unet1D
from dphm_tum.models.diffusion_tracking import inference_identity_fitting, inference_incremental_expression, inference_joint_chunk_fitting

from dphm_tum.utils.io import export_pointcloud_o3d
from dphm_tum.utils.pyvista_render import calcaute_nearsest_distance, render_snapshot, render_from_camera, blend_color_and_recon, read_intrinsics_kinect_from_json, transform_points_world2cam
from dphm_tum.utils.kinect import delete_nan_and_inside_kinect, random_sample, add_gaussian_noise
from dphm_tum.utils.landmark import IBU68_index_into_WFLW, ANCHOR_iBUG68_pairs_39, MEDIA_PIPE_MOUTH_EYES_index, FLAME_MOUTH_EYES_index
from dphm_tum.utils.load_checkpoint import load_pretrained_nphm_backward, load_pretrained_identity_and_expression_diffusion
from utils import load_tracking_parameters, save_tracking_parameters, transform_points_from_nphm_to_scan_space, transform_points_from_scan_to_nphm_space

parser = argparse.ArgumentParser(
    description='Run generation'
)

parser.add_argument('-cfg_file', type=str, default='./scripts/tracker/dphm_ncos.05_a.1_exprsmo1_rigsmo1_sds.5.yaml')
parser.add_argument('-exp_tag', type=str, required=True)
# different stages/strategy of point cloud sequence fitting
parser.add_argument('-init', required=False, action='store_true')
parser.add_argument('-incre', required=False, action='store_true')
parser.add_argument('-joint', required=False, action='store_true')
parser.add_argument('-gen', required=False, action='store_true')
# some options for data preprocess and loss function 
parser.add_argument('-subj', type=str, required=False)
parser.add_argument('-simple', required=False, action='store_true')
## some options during saving results
parser.add_argument('-save_flame', required=False, action='store_true')
parser.add_argument('-save_lms3d', required=False, action='store_true')
parser.add_argument('-save_scan_nphm', required=False, action='store_true')
parser.add_argument('-retrieve', required=False, action='store_true')
args = parser.parse_args()

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]
    
with open(args.cfg_file, 'r') as f:
    print('Loading config file from: ' + args.cfg_file)
    CFG = yaml.safe_load(f)
print(json.dumps(CFG, sort_keys=True, indent=4))
config = OmegaConf.create(CFG)

##################### load pretrained nphm backward model
device = torch.device("cuda")
neural_3dmm, latent_codes = load_pretrained_nphm_backward(config)

###################### load shape, expression diffusion
diffusion_shape, diffusion_expre = load_pretrained_identity_and_expression_diffusion(config, device, latent_codes)
##########################


anchor_path = env_paths.ANCHOR_INDICES_PATH.format(config.nphm_backward.num_anchors) 
lm_inds = np.load( anchor_path )
#anchors = torch.from_numpy( lm_inds ).float().unsqueeze(0).unsqueeze(0).to(device)

# create output directory
out_dir = env_paths.EXPERIMENT_DIR + '/dphm_kinect_tracking/{}/'.format(args.exp_tag)
os.makedirs(out_dir, exist_ok=True)
fname = out_dir + 'configs.yaml'
with open(fname, 'w') as yaml_file:
    yaml.safe_dump(CFG, yaml_file, default_flow_style=False)

def kinect_depth_tracking():
    np.random.seed(0)
    torch.manual_seed(0)
    print('############ Starting Fitting ############')

    cfg_data = config.data.kinect
    data_dir = cfg_data.data_dir
    flame_fit_dir = cfg_data.flame_fit_dir
    color_foldername = cfg_data.color_foldername
    depth_foldername = cfg_data.depth_foldername
    depth_postfix = cfg_data.depth_postfix
    pc_crop_foldername = cfg_data.pc_crop_foldername
    nms_crop_foldername = cfg_data.nms_crop_foldername
    lms_pip_foldername= cfg_data.lms_pip_foldername
    lms_media_pipe_foldername = cfg_data.lms_media_pipe_foldername
    param_filename = cfg_data.param_filename
    flame_foldername = cfg_data.flame_foldername
    use_quaternion = cfg_data.use_quaternion
    convention = cfg_data.convention
    flame2scan = cfg_data.get('flame2scan', True)
    resolution = cfg_data.get('resolution', 256)
    batch_points = cfg_data.get('batch_points', 100000)
    error_max  = cfg_data.get('error_max', 0.006)
    sample     = cfg_data.get('sample', False)
    num_points = cfg_data.get('num_points', 100000)
    noise_level= cfg_data.get('noise_level', 0)
    height     = cfg_data.get('height', 1080)
    width      = cfg_data.get('width', 1920)
    pipnet_lms = cfg_data.get('pipnet_lms', False)
    raw        = cfg_data.get('raw', False)
    raw2       = cfg_data.get('raw2', True)
    # grid points for marching cubes
    mini = [-.55, -.5, -.95]
    maxi = [0.55, 0.75, 0.4]
    grid_points = create_grid_points_from_bounds(mini, maxi, resolution)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    
    # select the tracking sequences
    if args.subj is not None:
        subjects = [ args.subj ]
    elif cfg_data.subjects is not None:
        subjects =  cfg_data.subjects
    else:
        subjects =  sorted(os.listdir( cfg_data.data_dir ))
    print("subject names :", subjects)
        
    if args.init:
        out_stage_dir = osp.join(out_dir, 'init')
    elif args.incre:
        out_stage_dir = osp.join(out_dir, 'incre')
    elif args.joint:
        out_stage_dir = osp.join(out_dir, 'joint')
    
    for subj in subjects:
        print('Fitting subject {}'.format(subj))
        flame_fit_param_filepath = osp.join(flame_fit_dir, subj, param_filename)
        if not os.path.exists(flame_fit_param_filepath):
            print(f"skipping the identity {subj} as we dont have the flame fitting parameters")
            continue
        out_stage_subj_dir = osp.join(out_stage_dir, subj)
        os.makedirs(out_stage_subj_dir, exist_ok=True)
        out_lat_dir = osp.join(out_dir, 'latent')
        os.makedirs(out_lat_dir, exist_ok=True)
        
        all_fns = [fns for fns in os.listdir(osp.join(data_dir, subj, pc_crop_foldername)) if fns.endswith('.npy')]
        # height, width = 1080, 1920
        intrinsics = read_intrinsics_kinect_from_json(osp.join(data_dir, subj, 'calibration.json'), (width, height), crop_details=None)
            
        number_of_frames = len( all_fns )
        all_color_path = []
        all_depth_path = []
        all_points_exp = []
        all_points_exp_dense = []
        all_points_exp_nphm = []
        all_normals_exp = []
        all_normals_exp_dense = []
        all_normals_exp_nphm = []
        all_lms_pip_exp = []
        all_lms_exp = []
        all_lms_exp_nphm = []
        allR = []
        allReuler = []
        allT = []
        allS = []
        all_flame_trimesh_nphm = []
        all_flame_anchors_nphm = []
        
        if args.init:
            keyframe = CFG['fitting_init']['keyframe']
        elif args.incre:
            keyframe = CFG['fitting_incre']['keyframe']
        elif args.joint:
            keyframe = CFG['fitting_joint']['keyframe']
        else:
            keyframe = False
        if keyframe:
            frame_joint_list = range(0, 1)
        else:
            frame_joint_list = range(number_of_frames) 
            
        for i, frame in enumerate(frame_joint_list):
            print('Fitting subject {}, frame {}'.format(subj, frame))
            color_path = osp.join(data_dir, subj, '{:s}/{:05d}.png'.format(color_foldername, frame))
            depth_path = osp.join(data_dir, subj, '{:s}/{:05d}_{:s}'.format(depth_foldername, frame, depth_postfix))
            point_cloud_np = np.load(osp.join(data_dir, subj, '{:s}/{:05d}.npy'.format(pc_crop_foldername, frame)))
            normals_np = np.load(osp.join(data_dir, subj, '{:s}/{:05d}.npy'.format(nms_crop_foldername, frame)))
            # !!!!! flipp it back to camera as we use sum loss 
            normals_np = - normals_np
            paras_mvs2flame = np.load( osp.join(flame_fit_dir, subj, param_filename) )
            R = paras_mvs2flame['rotation_matrices'][frame]  
            t = paras_mvs2flame['translation'][frame].squeeze()
            c = paras_mvs2flame['scale'].squeeze()
            flame_trimesh = trimesh.load( osp.join(flame_fit_dir, subj, flame_foldername, '{:05d}.ply'.format(frame)), process=False )
        
            if pipnet_lms:
                lms_np = np.load(osp.join(data_dir, subj, '{:s}/{:05d}.npy'.format(lms_pip_foldername, frame)))
                order = ANCHOR_iBUG68_pairs_39[:, 0]
                pip_idx = IBU68_index_into_WFLW[ANCHOR_iBUG68_pairs_39[:, 1]]
                lms = lms_np[ pip_idx[:], :]
            else:
                lms_np = np.load(osp.join(data_dir, subj, '{:s}/{:05d}.npy'.format(lms_media_pipe_foldername, frame)))
                lms = lms_np[MEDIA_PIPE_MOUTH_EYES_index, :]  
                order = FLAME_MOUTH_EYES_index 
            
            print(i, point_cloud_np.shape, normals_np.shape, pc_crop_foldername, nms_crop_foldername)
            # delete nan and convert raw points/normals/anchors to the nphm coodinate system
            # select points within a range in nphm coodrinate system
            point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm = delete_nan_and_inside_kinect(point_cloud_np, normals_np, R, t, c, flame2scan=flame2scan, raw_scan=raw, raw_scan2=raw2)
    
            point_cloud_dense = point_cloud_np
            normals_dense = normals_np
                
            if sample:
                point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm \
                    = random_sample(point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm, num_points=num_points)
            
            if noise_level > 0:
                point_cloud_np, normals_np = add_gaussian_noise(point_cloud_np, normals_np, noise_level=noise_level)
            
            
            # get flame mesh anchors and replace some with 3d landmarks
            flame_trimesh_nphm = flame_trimesh.copy()
            flame_trimesh_nphm.vertices *=4.0
            flame_anchors_nphm = flame_trimesh_nphm.vertices[lm_inds, :]
            lms_nphm = transform_points_from_scan_to_nphm_space(lms, c, R, t)
            
            #replace flame anchors with anchors from depth/scans
            flame_anchors_nphm[order, :] = lms_nphm
            lms =  torch.from_numpy(lms).float().to(device)
            lms_nphm = torch.from_numpy(lms_nphm).float().to(device)
            flame_anchors_nphm_torch = torch.from_numpy(flame_anchors_nphm).float().to(device)
            
            # convert it to tensors
            point_cloud_torch = torch.from_numpy(point_cloud_np).float().to(device)
            normals_torch = torch.from_numpy(normals_np).float().to(device)
            point_cloud_nphm_torch = torch.from_numpy(point_cloud_np_nphm).float().to(device)
            normals_nphm_torch = torch.from_numpy(normals_np_nphm).float().to(device)
            point_cloud_dense_torch = torch.from_numpy(point_cloud_dense).float().to(device)
            normals_dense_torch = torch.from_numpy(normals_dense).float().to(device)
            
            all_color_path.append( color_path )
            all_depth_path.append( depth_path )
            all_points_exp.append( point_cloud_torch )
            all_points_exp_dense.append( point_cloud_dense_torch )
            all_points_exp_nphm.append( point_cloud_nphm_torch)
            all_normals_exp.append( normals_torch )
            all_normals_exp_dense.append( normals_dense_torch )
            all_normals_exp_nphm.append( normals_nphm_torch )
            all_lms_pip_exp.append( lms_np )
            all_lms_exp.append( lms )
            all_lms_exp_nphm.append( lms_nphm )
            if use_quaternion:
                allReuler.append(matrix_to_quaternion(torch.from_numpy(R)).to(device) )
            else:
                allReuler.append(matrix_to_euler_angles(torch.from_numpy(R), convention=convention).to(device) )
            allR.append( torch.from_numpy(R).float().to(device) )
            allT.append( torch.from_numpy(t).float().to(device) )
            allS.append( c )
            all_flame_trimesh_nphm.append( flame_trimesh_nphm )
            all_flame_anchors_nphm.append( flame_anchors_nphm_torch ) 

        sds_optim = CFG["sds_optim"]
        if args.init and not args.gen:
            print("optimize identity of the key frame!!!")
            cfg_fit = CFG['fitting_init'] 
            if args.simple:
                lambdas = cfg_fit['lambdas_simple']
            else:
                lambdas = cfg_fit['lambdas']
            n_steps = cfg_fit['n_steps']
            init_lr = cfg_fit["init_lr"]
            init_lr_rigid = cfg_fit["init_lr_rigid"]
            schedule_cfg = { 'lr': cfg_fit["lr_schedule"] }
            schedule_rigid_cfg = {'lr': cfg_fit["lr_rigid_schedule"]}
            
            # load inistial latent codes of individual frame fitting    
            init_lat_file = osp.join(out_lat_dir, subj + "_init.npz")
            if os.path.exists(init_lat_file):
                lat_reps_expr, lat_rep_shape, trans, rot, rot_euler, scale = load_tracking_parameters(init_lat_file, device)
                allReuler_init = [ rot_euler[rid] for rid in range(rot_euler.shape[0]) ]
                allT_init = [ trans[tid] for tid in range(trans.shape[0]) ]
                allS_init =  [ scale.cpu().numpy() for rid in range(rot_euler.shape[0]) ]
                print("read previously optimized parametes from :", init_lat_file)
            else:
                lat_rep_shape = torch.zeros((1, 1, latent_codes.codebook['geo'].dim)).float().to(device)
                lat_reps_expr =  torch.zeros((len(all_points_exp), 1, latent_codes.codebook['exp'].dim)).float().to(device)
                allReuler_init = allReuler
                allT_init = allT
                allS_init = allS
                
            lat_reps_expr, lat_rep_shape, anchors, rigid_transform_dict = inference_identity_fitting(
                                                                                    neural_3dmm,
                                                                                    diffusion_shape, diffusion_expre, 
                                                                                    lat_reps_expr, lat_rep_shape, 
                                                                                    lambdas, sds_optim,
                                                                                    all_points_exp, all_normals_exp,
                                                                                    allReuler_init, allT_init, allS_init, all_lms_exp,
                                                                                    order, init_lr, init_lr_rigid, n_steps,
                                                                                    schedule_cfg,  schedule_rigid_cfg, 
                                                                                    calculate_normal=cfg_fit["cal_normal"], 
                                                                                    calculate_anchor=cfg_fit["cal_anchor"],
                                                                                    opt_iden=cfg_fit["opt_iden"],
                                                                                    opt_expre=cfg_fit["opt_expre"],
                                                                                    opt_rigid_scale=cfg_fit["opt_rigid_scale"],
                                                                                    log_loss=cfg_fit["log_loss"],
                                                                                    project=CFG["project"], exp_name=cfg_fit["exp_name"],
                                                                                    use_quaternion = use_quaternion, 
                                                                                    convention = convention,
                                                                                    flame2scan = flame2scan,
                                                                                    )
            
            # save inistial latent codes of individual frame fitting 
            init_lat_file = osp.join(out_lat_dir, subj + "_init.npz")
            save_tracking_parameters(init_lat_file, lat_reps_expr, lat_rep_shape, rigid_transform_dict)
            print("save optimized tracking parametes to:", init_lat_file)

        
        if args.incre and not args.gen:
            print("optimize nonrigid parameters frame by frame")
            cfg_fit = CFG["fitting_incre"]
            if args.simple:
                lambdas = cfg_fit['lambdas_simple']
            else:
                lambdas = cfg_fit['lambdas']
            init_lr  = cfg_fit["init_lr"]
            init_lr2 = cfg_fit["init_lr2"]
            init_lr_rigid = cfg_fit["init_lr_rigid"]
            n_steps = cfg_fit["n_steps"]
            n_steps2 = cfg_fit["n_steps2"]

            schedule_cfg = { 'lr': cfg_fit["lr_schedule"] }
            schedule2_cfg = { 'lr': cfg_fit["lr2_schedule"] }
            schedule_rigid_cfg = { 'lr': cfg_fit["lr_rigid_schedule"] }

            # load inistial latent codes of individual frame fitting    
            init_lat_file = osp.join(out_lat_dir, subj + "_init.npz")
            lat_reps_expr_init, lat_rep_shape, trans, rot, rot_euler, scale = load_tracking_parameters(init_lat_file, device)
            lat_reps_expr =  torch.concat([lat_reps_expr_init, \
                lat_reps_expr_init[-1:, :, :].repeat(len(all_points_exp)-lat_reps_expr_init.shape[0], 1, 1)], dim=0).contiguous()
                 
            # calcualte delta roteuler and trans
            rot_euler_delta = torch.stack([ rot_euler[rid] - allReuler[rid]  for rid in range(rot_euler.shape[0]) ], dim=0).mean(dim=0)
            trans_delta = torch.stack([ trans[tid]-allT[tid] for tid in range(trans.shape[0]) ], dim=0).mean(dim=0)
            
            # get corrected initialization for incremental refinement
            allReuler_incre = [ rot_euler[rid] for rid in range(rot_euler.shape[0]) ] +  [ allReuler[rid] + rot_euler_delta  for rid in range(rot_euler.shape[0], len(allReuler)) ]   
            allT_incre = [ trans[tid] for tid in range(trans.shape[0]) ] + [ allT[tid] + trans_delta  for tid in range(trans.shape[0], len(allT)) ]
            allS_incre =  [ scale.cpu().numpy() for rid in range(len(all_points_exp)) ]
            print('add corrective transformation to all other frames !!!!!', rot_euler_delta.cpu().numpy(), trans_delta.cpu().numpy())
            
            
            
            lat_reps_expr, lat_rep_shape, _, rigid_transform_dict = inference_incremental_expression(
                                                                                        neural_3dmm,
                                                                                        diffusion_shape, diffusion_expre,
                                                                                        lat_reps_expr, lat_rep_shape, 
                                                                                        lambdas, sds_optim,
                                                                                        all_points_exp, all_normals_exp,
                                                                                        allReuler_incre, allT_incre, allS_incre, all_lms_exp,
                                                                                        order, init_lr, init_lr2, init_lr_rigid, n_steps, n_steps2,
                                                                                        schedule_cfg, schedule2_cfg,  schedule_rigid_cfg, 
                                                                                        calculate_normal=cfg_fit["cal_normal"], 
                                                                                        calculate_anchor=cfg_fit["cal_anchor"],
                                                                                        opt_rigid_scale=cfg_fit["opt_rigid_scale"], 
                                                                                        opt_iden=cfg_fit["opt_iden"],
                                                                                        opt_expre=cfg_fit["opt_expre"],
                                                                                        log_loss=cfg_fit["log_loss"],
                                                                                        project=CFG["project"], exp_name=cfg_fit["exp_name"],
                                                                                        use_quaternion = use_quaternion, 
                                                                                        convention = convention,
                                                                                        flame2scan = flame2scan
                                                                                        )

            # save inistial latent codes of individual frame fitting 
            incre_lat_file = osp.join(out_lat_dir, subj + "_incre.npz")
            save_tracking_parameters(incre_lat_file, lat_reps_expr, lat_rep_shape, rigid_transform_dict)
            print("save optimized tracking parametes to:", incre_lat_file)

    
        # finetune both rigid and non-rigid registration parameters:
        if args.joint and not args.gen:
            print("joint nin-rigid and rigid registration")
            cfg_fit = CFG["fitting_joint"]
            if args.simple:
                lambdas = cfg_fit['lambdas_simple']
            else:
                lambdas = cfg_fit['lambdas']
            init_lr = cfg_fit["init_lr"]
            init_lr_rigid = cfg_fit["init_lr_rigid"]
            n_steps = cfg_fit["n_steps"]

            schedule_cfg = { 'lr': cfg_fit['lr_schedule'] }
            schedule_rigid_cfg = {'lr': cfg_fit['lr_rigid_schedule']}
            
            # load latent codes of individual frame fitting    
            incre_lat_file = osp.join(out_lat_dir, subj + "_incre.npz")
            lat_reps_expr_incre, lat_rep_shape, trans,  rot, rot_euler, scale = load_tracking_parameters(incre_lat_file, device)
            allReuler_joint = [ rot_euler[rid] for rid in range(rot_euler.shape[0]) ]
            allT_joint = [ trans[tid] for tid in range(trans.shape[0]) ]
            allS_joint =  [ scale.cpu().numpy() for rid in range(rot_euler.shape[0]) ]
            
            lat_reps_expr, lat_rep_shape, anchors, rigid_transform_dict = inference_joint_chunk_fitting(
                                                                                    neural_3dmm,
                                                                                    diffusion_shape, diffusion_expre, 
                                                                                    lat_reps_expr_incre, lat_rep_shape, 
                                                                                    lambdas, sds_optim,
                                                                                    all_points_exp, all_normals_exp,
                                                                                    allReuler_joint, allT_joint, allS_joint, all_lms_exp,
                                                                                    order, init_lr, init_lr_rigid, n_steps,
                                                                                    schedule_cfg,  schedule_rigid_cfg, 
                                                                                    calculate_normal=cfg_fit["cal_normal"], 
                                                                                    calculate_anchor=cfg_fit["cal_anchor"],
                                                                                    opt_rigid=cfg_fit["opt_rigid"], 
                                                                                    opt_scale=cfg_fit["opt_scale"], 
                                                                                    opt_iden=cfg_fit["opt_iden"],
                                                                                    opt_expre=cfg_fit["opt_expre"],
                                                                                    log_loss=cfg_fit["log_loss"],
                                                                                    project=CFG["project"], exp_name=cfg_fit["exp_name"],
                                                                                    use_quaternion = use_quaternion, 
                                                                                    convention = convention,
                                                                                    flame2scan = flame2scan,
                                                                                    )
        
            # save the latent codes of joint frame fitting
            joint_lat_file = osp.join(out_lat_dir, subj + "_joint.npz")
            save_tracking_parameters(joint_lat_file, lat_reps_expr, lat_rep_shape, rigid_transform_dict)
            print("save optimized tracking parametes to:", joint_lat_file)
            
        if args.init:
            init_lat_file = osp.join(out_lat_dir, subj + "_init.npz")
            if os.path.exists(init_lat_file):
                lat_reps_expr, lat_rep_shape, trans, rot, rot_euler, scale = load_tracking_parameters(init_lat_file, device)
                print("read previously optimized paqrametes from :", init_lat_file)
        
        if args.incre:
            # load inistial latent codes of individual frame fitting    
            incre_lat_file = osp.join(out_lat_dir, subj + "_incre.npz")
            if os.path.exists(incre_lat_file):
                lat_reps_expr, lat_rep_shape, trans, rot, rot_euler, scale = load_tracking_parameters(incre_lat_file, device)
                print("read previously optimized parametes from :", incre_lat_file)   
        
        if args.joint:
            # load  the latent codes of joint frame fitting    
            joint_lat_file = osp.join(out_lat_dir, subj + "_joint.npz")
            if os.path.exists(joint_lat_file):
                lat_reps_expr, lat_rep_shape, trans, rot, rot_euler, scale = load_tracking_parameters(joint_lat_file, device)
                print("read previously optimized parametes from :", joint_lat_file)

        encoding_expr_zero = torch.zeros_like( latent_codes.codebook['exp'].embedding.weight.data[0:1, None, :] ).float().to(device)
        condition = { 'geo': lat_rep_shape, 'exp': encoding_expr_zero }
        logits = get_logits(neural_3dmm, condition, grid_points.clone(), nbatch_points=batch_points)
        mesh_can = mesh_from_logits(logits, mini, maxi, resolution)
        mesh_can.export( osp.join(out_stage_subj_dir, '{}_neutral.ply'.format(subj) ) )
        if args.retrieve:
            shape_dist = torch.sum(( latent_codes.codebook['geo'].embedding.weight.data[:, None, :] - lat_rep_shape )**2, dim=[1, 2])
            shape_idx = torch.argmin(shape_dist)
            lat_rep_shape_retrive = latent_codes.codebook['geo'].embedding.weight.data[shape_idx:shape_idx+1, None, :]
            condition = { 'geo': lat_rep_shape_retrive, 'exp': encoding_expr_zero }
            logits = get_logits(neural_3dmm, condition, grid_points.clone(), nbatch_points=batch_points)
            mesh_can_retrieve = mesh_from_logits(logits, mini, maxi, resolution)
            mesh_can_retrieve.export(osp.join(out_stage_subj_dir, '{}_neutral_retrieve.ply'.format(subj) ) )
        
        result_mesh_dir = osp.join(out_stage_subj_dir, "result")    
        result_snapshot_blend_dir = osp.join(out_stage_subj_dir, "result", "blend")
        result_snapshot_render_dir = osp.join(out_stage_subj_dir, "result", "render")
        os.makedirs(result_mesh_dir, exist_ok=True)
        os.makedirs(result_snapshot_blend_dir, exist_ok=True)
        os.makedirs(result_snapshot_render_dir, exist_ok=True)
        error_snapshot_dir = osp.join(out_stage_subj_dir, "result", "error")
        os.makedirs(error_snapshot_dir, exist_ok=True)        
        
        result_mesh_dir = osp.join(out_stage_subj_dir, "result")    
        result_snapshot_blend_dir = osp.join(out_stage_subj_dir, "result", "blend")
        result_snapshot_render_dir = osp.join(out_stage_subj_dir, "result", "render")
        os.makedirs(result_mesh_dir, exist_ok=True)
        os.makedirs(result_snapshot_blend_dir, exist_ok=True)
        os.makedirs(result_snapshot_render_dir, exist_ok=True)
        error_snapshot_dir = osp.join(out_stage_subj_dir, "result", "error")
        os.makedirs(error_snapshot_dir, exist_ok=True)  
        
        ################
        scan_pc_dir = osp.join(out_stage_subj_dir, "scan")
        scan_snapshot_blend_dir = osp.join(out_stage_subj_dir, "scan", "blend")
        scan_snapshot_blend_nm_dir = osp.join(out_stage_subj_dir, "scan", "blend_nm")
        scan_snapshot_render_dir = osp.join(out_stage_subj_dir, "scan", "render")
        scan_snapshot_render_nm_dir = osp.join(out_stage_subj_dir, "scan", "render_nm")
        scan_snapshot_error_dir = osp.join(out_stage_subj_dir, "scan", "error")
        os.makedirs(scan_pc_dir, exist_ok=True)
        os.makedirs(scan_snapshot_blend_dir, exist_ok=True)
        os.makedirs(scan_snapshot_blend_nm_dir, exist_ok=True)
        os.makedirs(scan_snapshot_render_dir, exist_ok=True)
        os.makedirs(scan_snapshot_render_nm_dir, exist_ok=True)
        os.makedirs(scan_snapshot_error_dir, exist_ok=True)
        
        if args.save_lms3d:
            scan_snapshot_lms3d_dir = osp.join(out_stage_subj_dir, "scan", "lms")
            os.makedirs(scan_snapshot_lms3d_dir, exist_ok=True)
            
        if args.save_flame:    
            flame_mesh_dir = osp.join(out_stage_subj_dir, "flame")
            os.makedirs(flame_mesh_dir, exist_ok=True)   
            flame_snapshot_error_dir = osp.join(out_stage_subj_dir, "flame", "error")
            flame_snapshot_scan_error_dir = osp.join(out_stage_subj_dir, "flame", "scan_error")
            flame_snapshot_render_dir = osp.join(out_stage_subj_dir, "flame", "render")
            os.makedirs(flame_snapshot_error_dir, exist_ok=True) 
            os.makedirs(flame_snapshot_scan_error_dir, exist_ok=True) 
            os.makedirs(flame_snapshot_render_dir, exist_ok=True) 
        
        for i, expr_ind in enumerate(frame_joint_list):
            # raw parameters of flame fitting
            rot_mat_raw, t_raw, c_raw = allR[i].detach().cpu().numpy(), allT[i].detach().cpu().numpy()[None, :], allS[i]
            rot_mat, t, c = rot[i].cpu().numpy(), trans[i][None, :].cpu().numpy(), scale.cpu().numpy()

            #################### input
            # save raw kinect scan sequences
            if all_points_exp[i].cpu().numpy().shape[0] > 10000:
                rand_idx = np.random.randint(0, all_points_exp[i].cpu().numpy().shape[0], size=10000) 
            else:
                rand_idx = np.arange(all_points_exp[i].cpu().numpy().shape[0])
            export_pointcloud_o3d(osp.join(scan_pc_dir, '{}_{}_scan_nm1w.ply'.format(subj, expr_ind)), all_points_exp[i].cpu().numpy()[rand_idx], normals=-all_normals_exp[i].cpu().numpy()[rand_idx])
            # render
            mesh_scan_snapshot_path = osp.join(scan_snapshot_render_dir, '{}_{}_scan.jpg'.format(subj, expr_ind))
            _ = render_from_camera(mesh_scan_snapshot_path, intrinsics=intrinsics, points=all_points_exp[i].cpu().numpy(), anchors=all_lms_exp[i].cpu().numpy(), vis_points=True, vis_anchors=True)
            # blend
            mesh_scan_blend_path = osp.join(scan_snapshot_blend_dir, '{}_{}_scan_blend.jpg'.format(subj, expr_ind))
            blend_color_and_recon(all_color_path[i], mesh_scan_snapshot_path, mesh_scan_blend_path)
            # render input scans with normals and all media pipeline anchors
            mesh_scan_normal_snapshot_path = osp.join(scan_snapshot_render_nm_dir, '{}_{}_scan_nm.jpg'.format(subj, expr_ind))
            _ = render_from_camera(mesh_scan_normal_snapshot_path, intrinsics=intrinsics, points=all_points_exp[i].cpu().numpy(), anchors=all_lms_pip_exp[i], normals=-all_normals_exp[i].cpu().numpy(), vis_points=True)
            mesh_scan_blend_nm_path = osp.join(scan_snapshot_blend_nm_dir, '{}_{}_scan_nm_blend.jpg'.format(subj, expr_ind))
            blend_color_and_recon(all_color_path[i], mesh_scan_normal_snapshot_path, mesh_scan_blend_nm_path)

            if args.save_scan_nphm:
                # save kinect scan sequence aligned with nphm
                mesh_scan_nphm = pv.PolyData(all_points_exp_nphm[i].cpu().numpy())
                mesh_scan_nphm.save(osp.join(scan_pc_dir, '{}_{}_scan_nphm.ply'.format(subj, expr_ind)))
                rand_idx = np.random.randint(0, all_points_exp_nphm[i].cpu().numpy().shape[0], size=10000)  
                export_pointcloud_o3d(osp.join(scan_pc_dir, '{}_{}_scan_nphm_nm1w.ply'.format(subj, expr_ind)), all_points_exp_nphm[i].cpu().numpy()[rand_idx], normals=-all_normals_exp_nphm[i].cpu().numpy()[rand_idx])
                render_snapshot(scan_snapshot_render_dir + '/{:05d}.jpg'.format(i), points=all_points_exp_nphm[i].cpu().numpy(), normals=-all_normals_exp_nphm[i].cpu().numpy(), vis_points=True, nphm_coord=True)
            ##################### input
            
            ##################### output
            condition_i = { 'geo': lat_rep_shape, 'exp': lat_reps_expr[i, ...].unsqueeze(0) }
            logits = get_logits(neural_3dmm, condition_i, grid_points.clone(), nbatch_points=batch_points)
            mesh = mesh_from_logits(logits, mini, maxi, resolution)
            mesh.vertices = transform_points_from_nphm_to_scan_space(mesh.vertices, c, rot_mat, t, flame2scan=flame2scan)
            
            mesh.export(osp.join(result_mesh_dir, '{}_{}.ply'.format(subj, expr_ind)))
            mesh_verts_dist = calcaute_nearsest_distance(np.array(mesh.vertices), all_points_exp_dense[i].cpu().numpy(), delta_dist=error_max)
            print(i,  "mesh vert dist", mesh_verts_dist.min(), mesh_verts_dist.max())
            _ = render_from_camera( error_snapshot_dir + '/{:05d}_max{:03f}.jpg'.format(expr_ind, error_max), intrinsics=intrinsics, mesh=mesh, mesh_errors=mesh_verts_dist, vis_mesh=True)
            
            scan_dist = calcaute_nearsest_distance(all_points_exp_dense[i].cpu().numpy(), np.array(mesh.vertices), delta_dist=error_max)
            print(i,  "scan dist", mesh_verts_dist.min(), mesh_verts_dist.max())
            _ = render_from_camera(scan_snapshot_error_dir+ '/{:05d}_max{:03f}.jpg'.format(expr_ind, error_max), intrinsics=intrinsics, points=all_points_exp_dense[i].cpu().numpy(), points_errors=scan_dist, vis_points=True)
            
            # render
            mesh_snapshot_path = osp.join(result_snapshot_render_dir, '{}_{}.jpg'.format(subj, expr_ind))
            _ = render_from_camera(mesh_snapshot_path, intrinsics=intrinsics,  mesh=mesh, anchors=None, vis_mesh=True, vis_anchors=False)
            # blend
            mesh_blend_path = osp.join(result_snapshot_blend_dir, '{}_{}_blend.jpg'.format(subj, expr_ind))
            blend_color_and_recon(all_color_path[i], mesh_snapshot_path, mesh_blend_path)
        
            if args.save_lms3d:
                mesh_scan_anchor_snapshot_path = osp.join(scan_snapshot_lms3d_dir, '{}_{}_lms_anc.jpg'.format(subj, expr_ind))
                _ = render_from_camera(mesh_scan_anchor_snapshot_path, intrinsics=intrinsics, anchors=all_lms_pip_exp[i], labels=[str(aaa) for aaa in range(all_lms_pip_exp[i].shape[0])], vis_anchors=True)
                # used eight landmarks
                anchors_input = pv.PolyData(all_lms_pip_exp[i])
                anchors_input.save(osp.join(scan_pc_dir, '{}_{}_lms3d_input.ply'.format(subj, expr_ind)) )

                # flame registrate anchors                
                flame_anchors_nphm_numpy = all_flame_anchors_nphm[i].cpu().numpy()
                flame_anchors_input_numpy = transform_points_from_nphm_to_scan_space(flame_anchors_nphm_numpy, c, rot_mat, t, flame2scan=flame2scan)    
                    
                flame_anchors_nphm_mesh = pv.PolyData(flame_anchors_input_numpy)
                flame_anchors_nphm_mesh.save(osp.join(scan_pc_dir, '{}_{}_flame_anc_input.ply'.format(subj, expr_ind)) )
                flame_anchors_nphm_snapshot_path = osp.join(scan_snapshot_lms3d_dir, '{}_{}_flame_anc.jpg'.format(subj, expr_ind))
                _ = render_from_camera(flame_anchors_nphm_snapshot_path, intrinsics=intrinsics, anchors=flame_anchors_input_numpy, labels=[str(aaa) for aaa in range(flame_anchors_input_numpy.shape[0])], vis_anchors=True)
        
            if args.save_flame:    
                # save mesh registration gt
                mesh_regi = all_flame_trimesh_nphm[i]
                mesh_regi.vertices = transform_points_from_nphm_to_scan_space(mesh_regi.vertices, c_raw, rot_mat_raw, t_raw, flame2scan=flame2scan)                  
                mesh_regi.export(osp.join(flame_mesh_dir, '{}_{}_flame_regi.ply'.format(subj, expr_ind)))
                mesh_regi_verts_dist = calcaute_nearsest_distance(np.array(mesh_regi.vertices), all_points_exp_dense[i].cpu().numpy(), delta_dist=error_max)
                print(i, "mesh regi vert dist", mesh_regi_verts_dist.min(), mesh_regi_verts_dist.max())             
                
                flame_snapshot_error_path = osp.join(flame_snapshot_error_dir, '{}_{}_max{:03f}.jpg'.format(subj, expr_ind, error_max))
                _ = render_from_camera(flame_snapshot_error_path, intrinsics=intrinsics, mesh=mesh_regi, mesh_errors=mesh_regi_verts_dist, vis_mesh=True)
                
                scan_dist = calcaute_nearsest_distance(all_points_exp_dense[i].cpu().numpy(), np.array(mesh_regi.vertices), delta_dist=error_max)
                print(i,  "scan 2 flame dist", scan_dist.min(), scan_dist.max())
                flame_snapshot_scan_error_path =  osp.join( flame_snapshot_scan_error_dir, '{}_{}_max{:03f}.jpg'.format(subj, expr_ind, error_max))
                _ = render_from_camera(flame_snapshot_scan_error_path, intrinsics=intrinsics, points=all_points_exp_dense[i].cpu().numpy(), points_errors=scan_dist, vis_points=True)  
                
                flame_snapshot_render_path = osp.join(flame_snapshot_render_dir, '{:05d}.jpg'.format(expr_ind) )
                _ = render_from_camera(flame_snapshot_render_path, intrinsics=intrinsics, mesh=mesh_regi, vis_mesh=True)
            ###############
        torch.cuda.empty_cache()
        
        if args.incre or args.joint:  
            render_folder_list = [ result_snapshot_blend_dir, result_snapshot_render_dir, error_snapshot_dir,
                            scan_snapshot_blend_dir, scan_snapshot_blend_nm_dir, scan_snapshot_render_dir, scan_snapshot_render_nm_dir, scan_snapshot_error_dir]
                
            if args.save_flame:    
                render_folder_list += [flame_snapshot_error_dir, flame_snapshot_scan_error_dir, flame_snapshot_render_dir,]
                
            for folder_path in render_folder_list:
                video_filename = folder_path.split('/')[-1]
                os.system("python scripts/postprocess/create_video_from_img.py -i {:s} -f {:s} -ext {:s} -fps {:d} -ng".format(folder_path, video_filename, 'jpg', 15))
            

if __name__ == '__main__':
    kinect_depth_tracking()