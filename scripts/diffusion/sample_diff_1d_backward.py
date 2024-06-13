import argparse
import torch
import json, os, yaml
import torch
import numpy as np
import os.path as osp
from glob import glob

from omegaconf import OmegaConf
from dphm_tum import env_paths
from dphm_tum.models.denoising_diffusion_pytorch_1d import Trainer1D, GaussianDiffusion1D, Unet1D, render_snapshot
from dphm_tum.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from dphm_tum.models.reconstruction import get_logits
from dphm_tum.data.face_dataset import ScannerJson
from nphm_utils import load_pretrained_nphm_backward

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
parser.add_argument('-ckpt', required=False,  type=int)
parser.add_argument('-progress', required=False, action='store_true')
parser.add_argument('-id_path', required=False, type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))
config = OmegaConf.create(CFG)

exp_dir = env_paths.EXPERIMENT_DIR + '/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
device = torch.device("cuda")

# load pretrained nphm backward model
neural_3dmm, latent_codes = load_pretrained_nphm_backward(config)

_identity_embedding = latent_codes.codebook['geo'].embedding
_expression_embedding = latent_codes.codebook['exp'].embedding
_identity_latent_codes = latent_codes.codebook['geo'].embedding.weight.data
_expression_latent_codes = latent_codes.codebook['exp'].embedding.weight.data
print("identity latent code size of training set :", _identity_latent_codes.shape)
print("expression latent code size of training set :", _expression_latent_codes.shape)
_identity_latent_codes_min, _identity_latent_codes_max = _identity_latent_codes.min(), _identity_latent_codes.max()
_expression_latent_codes_min, _expression_latent_codes_max = _expression_latent_codes.min(), _expression_latent_codes.max()
print('training identity latent code min max :', _identity_latent_codes_min, _identity_latent_codes_max)
print('training expression latent code min max :', _expression_latent_codes_min, _expression_latent_codes_max)

_iden_lat_mean, _iden_lat_std =  _identity_latent_codes.mean(dim=0, keepdim=True), _identity_latent_codes.std()
_expr_lat_mean, _expr_lat_std = _expression_latent_codes.mean(dim=0, keepdim=True), _expression_latent_codes.std()
_joint_lat_mean, _joint_lat_std = torch.cat([_iden_lat_mean, _expr_lat_mean], dim=-1), max(_iden_lat_std, _expr_lat_std)
print('identity latent std :', _iden_lat_std)
print('expression latent std :', _expr_lat_std)
print('joint latent std :', _joint_lat_std)

if CFG['training']['diff_type'] == "shape":
    minimum = _identity_latent_codes_min
    maximum = _identity_latent_codes_max
    project = "shape_space_diffusion"
elif CFG['training']['diff_type'] == "expre":
    minimum = _expression_latent_codes_min
    maximum = _expression_latent_codes_max
    project = 'scanner_deformations_diffusion'
elif CFG['training']['diff_type'] == "joint":
    minimum = min(_identity_latent_codes_min, _expression_latent_codes_min)
    maximum = max(_identity_latent_codes_max, _expression_latent_codes_max)
    project = 'scanner_deformations_diffusion'
else:
    raise NotImplementedError

weight_dir_diff = env_paths.EXPERIMENT_DIR + '/{}/'.format(args.exp_name)
checkpoint_path = weight_dir_diff + 'checkpoints/'
checkpoints = glob(checkpoint_path+'/*')
checkpoints = [os.path.splitext(os.path.basename(path))[0][6:] for path in checkpoints]
checkpoints = np.array(checkpoints, dtype=int)
checkpoints = np.sort(checkpoints)
if args.ckpt:
    ckpt = args.ckpt
else:
    ckpt = checkpoints[-1]
print('LOADING', ckpt)
path = checkpoint_path + 'model-{}.pt'.format(ckpt)
print('Loaded checkpoint from: {}'.format(path))
data = torch.load(path, map_location=device)

model = Unet1D(
    **CFG['diffusion']['net_kwargs'],
)
model = model.to(device)

diffusion = GaussianDiffusion1D(
    model,
    **CFG['diffusion']['diff_kwargs'],
    minimum = minimum,
    maximum = maximum,
)
diffusion = diffusion.to(device)
diffusion.load_state_dict(data['model'], strict=True)

bbox_min = [-.55, -.5, -.95]
bbox_max = [.55, 0.75, 0.4]
res = 256
grid_points = create_grid_points_from_bounds(bbox_min, bbox_max, res)
grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)

method_name = 'dphm'
if args.progress:
    sample_mesh_path = weight_dir_diff + '{}_mesh_samples_progress_{}'.format(method_name, ckpt)
    os.makedirs(sample_mesh_path, exist_ok=True)
    sample_snapshot_path = weight_dir_diff + '{}_snapshot_samples_progress_{}'.format(method_name, ckpt)
    os.makedirs(sample_snapshot_path, exist_ok=True)

    if CFG['training']['diff_type'] == "expre" or CFG['training']['diff_type'] == "joint":
        sample_mesh_cano_path = weight_dir_diff + '{}_mesh_samples_progress_{}/cano'.format(method_name, ckpt)
        os.makedirs(sample_mesh_cano_path, exist_ok=True)
        sample_snapshot_cano_path = weight_dir_diff + '{}_snapshot_samples_progress_{}/cano'.format(method_name, ckpt)
        os.makedirs(sample_snapshot_cano_path, exist_ok=True)
        
        sample_mesh_expre_path = weight_dir_diff + '{}_mesh_samples_progress_{}/expre'.format(method_name, ckpt)
        os.makedirs(sample_mesh_expre_path, exist_ok=True)
        sample_snapshot_expre_path = weight_dir_diff + '{}_snapshot_samples_progress_{}/expre'.format(method_name, ckpt)
        os.makedirs(sample_snapshot_expre_path, exist_ok=True)
else:
    sample_mesh_path = weight_dir_diff + '{}_mesh_samples_{}'.format(method_name, ckpt)
    os.makedirs(sample_mesh_path, exist_ok=True)
    sample_snapshot_path = weight_dir_diff + '{}_snapshot_samples_{}'.format(method_name, ckpt)
    os.makedirs(sample_snapshot_path, exist_ok=True)

    if CFG['training']['diff_type'] == "expre" or CFG['training']['diff_type'] == "joint":
        sample_mesh_cano_path = weight_dir_diff + '{}_mesh_samples_{}/cano'.format(method_name, ckpt)
        os.makedirs(sample_mesh_cano_path, exist_ok=True)
        sample_snapshot_cano_path = weight_dir_diff + '{}_snapshot_samples_{}/cano'.format(method_name, ckpt)
        os.makedirs(sample_snapshot_cano_path, exist_ok=True)
        
        sample_mesh_expre_path = weight_dir_diff + '{}_mesh_samples_{}/expre'.format(method_name, ckpt)
        os.makedirs(sample_mesh_expre_path, exist_ok=True)
        sample_snapshot_expre_path = weight_dir_diff + '{}_snapshot_samples_{}/expre'.format(method_name, ckpt)
        os.makedirs(sample_snapshot_expre_path, exist_ok=True)

if args.progress:
    if CFG['training']['diff_type'] == "expre":
        iters = 5
        batch_size = 1
        step = 200
        start = 0
        #select = [19] #[2, 3, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19]
        encoding_shape = torch.from_numpy(np.load(args.id_path)['id']).float().to(device)
        with torch.no_grad():
            for i in range(start, start+iters):
                # if i not in select:
                #     continue
                sample_latents = diffusion.sample_intermediate(batch_size = batch_size, step=step)
                sample_latents = torch.stack(sample_latents, dim=1) # B x L x N x C
                print(sample_latents.shape, sample_latents.max(), sample_latents.min())
                
                #encoding_shape = _identity_latent_codes[i:i+1, None, :]
                encoding_expr_zero = torch.zeros(_expression_latent_codes[0:1, None, :].shape).float().to(device)
                condition = { 'geo': encoding_shape, 'exp': encoding_expr_zero }
                logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                mesh = mesh_from_logits(logits, bbox_min, bbox_max, res)
                mesh.export(sample_mesh_cano_path + '/{:03d}.ply'.format(i))  
                render_snapshot(mesh, sample_snapshot_cano_path + '/{:04d}.png'.format(i))

                for j in range(batch_size):
                    for k in range(sample_latents.shape[1]):
                        encoding_expr = sample_latents[j:j+1, k, :, :]
                        condition = { 'geo': encoding_shape, 'exp': encoding_expr }
                        logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                        deformed_mesh = mesh_from_logits(logits, bbox_min, bbox_max, res)
                        os.makedirs(sample_mesh_expre_path + '/{:03d}'.format(i), exist_ok=True)
                        deformed_mesh.export(sample_mesh_expre_path + '/{:03d}/{:03d}_{:04d}.ply'.format(i, j, k * step))  
                        print(deformed_mesh.vertices.shape, deformed_mesh.faces.shape)
                        os.makedirs(sample_snapshot_expre_path + '/{:04d}'.format(i), exist_ok=True)
                        render_snapshot(deformed_mesh, sample_snapshot_expre_path + '/{:04d}/{:04d}_{:04d}.png'.format(i, j, k * step))    
                
    else:
        iters = 6
        batch_size = 1
        step = 100
        with torch.no_grad():
            for i in range(iters):
                sample_latents = diffusion.sample_intermediate(batch_size = batch_size, step=step)
                sample_latents = torch.stack(sample_latents, dim=1) # B x L x N x C
                print(sample_latents.shape, sample_latents.max(), sample_latents.min())
                
                for j in range(batch_size):
                    for k in range(sample_latents.shape[1]):
                        if CFG['training']['diff_type'] == "shape":
                            print('identity diffusion sampling')
                            encoding_shape = sample_latents[j:j+1, k, :, :] 
                            
                            encoding_expr_zero = torch.zeros(_expression_latent_codes[0:1, None, :].shape).float().to(device) 
                            condition = { 'geo': encoding_shape, 'exp': encoding_expr_zero }
                            logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                            trim = mesh_from_logits(logits, bbox_min, bbox_max, res)

                            trim.export(sample_mesh_path + '/{:03d}_{:04d}.ply'.format(i*batch_size+j, k * step) )
                            print(trim.vertices.shape, trim.faces.shape)
                            render_snapshot(trim, sample_snapshot_path + '/{:04d}_{:04d}.png'.format(i*batch_size+j, k * step) )
            
                    np.savez( sample_mesh_path + '/{:03d}.npz'.format(i*batch_size+j), id=sample_latents[j:j+1, -1, :, :].detach().cpu().numpy() )
                        
else:
    if CFG['training']['diff_type'] == "expre":
        iters = 20
        batch_size = 20 #10
        start=100 #0
        #select = [19] #[2, 3, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19]
        with torch.no_grad():
            for i in range(start, start+iters):
                # if i not in select:
                #     continue
                sample_latents = diffusion.sample(batch_size = batch_size)
                print(sample_latents.shape, sample_latents.max(), sample_latents.min())
                
                encoding_shape = _identity_latent_codes[i:i+1, None, :]
                encoding_expr_zero = torch.zeros(_expression_latent_codes[0:1, None, :].shape).float().to(device)
                condition = { 'geo': encoding_shape, 'exp': encoding_expr_zero }
                logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                mesh = mesh_from_logits(logits, bbox_min, bbox_max, res)
                mesh.export(sample_mesh_cano_path + '/{:03d}.ply'.format(i))  
                render_snapshot(mesh, sample_snapshot_cano_path + '/{:04d}.png'.format(i))

                for j in range(batch_size):
                    encoding_expr = sample_latents[j:j+1, :, :]
                    condition = { 'geo': encoding_shape, 'exp': encoding_expr }
                    logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                    deformed_mesh = mesh_from_logits(logits, bbox_min, bbox_max, res)
                    os.makedirs(sample_mesh_expre_path + '/{:03d}'.format(i), exist_ok=True)
                    deformed_mesh.export(sample_mesh_expre_path + '/{:03d}/{:03d}.ply'.format(i, j))  
                    print(deformed_mesh.vertices.shape, deformed_mesh.faces.shape)
                    os.makedirs(sample_snapshot_expre_path + '/{:04d}'.format(i), exist_ok=True)
                    render_snapshot(deformed_mesh, sample_snapshot_expre_path + '/{:04d}/{:04d}.png'.format(i, j))    
                    
    else:
        iters = 20
        batch_size = 20
        with torch.no_grad():
            for i in range(iters):
                sample_latents = diffusion.sample(batch_size = batch_size)
                print(sample_latents.shape, sample_latents.max(), sample_latents.min())
                
                for j in range(batch_size):
                    if CFG['training']['diff_type'] == "shape":
                        print('identity diffusion sampling')
                        encoding_shape = sample_latents[j:j+1, :, :] #torch.zeros(_identity_latent_codes[0:1, None, :].shape).float().to(device) # _identity_latent_codes[j:j+1, None, :] 
                        encoding_expr_zero = torch.zeros(_expression_latent_codes[0:1, None, :].shape).float().to(device) #_expression_latent_codes[j:j+1, None, :] 
                        condition = { 'geo': encoding_shape, 'exp': encoding_expr_zero }
                        logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                        trim = mesh_from_logits(logits, bbox_min, bbox_max, res)

                        trim.export(sample_mesh_path + '/{:03d}.ply'.format(i*batch_size+j))  
                        print(trim.vertices.shape, trim.faces.shape)
                        render_snapshot(trim, sample_snapshot_path + '/{:04d}.png'.format(i*batch_size+j))  
                        
                    elif CFG['training']['diff_type'] == "joint":
                        lat_dim_shape = CFG['nphm_backward']['z_global_shape'] + (CFG['nphm_backward']['num_anchors'] + 1) * CFG['nphm_backward']['z_local_shape']
                        lat_dim_expre = CFG['nphm_backward']['z_sdf']
                            
                        encoding_shape = sample_latents[j:j+1, :, :lat_dim_shape]
                        encoding_expr_zero = torch.zeros(_expression_latent_codes[0:1, None, :].shape).float().to(device)
                        condition = { 'geo': encoding_shape, 'exp': encoding_expr_zero }
                        logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                        mesh = mesh_from_logits(logits, bbox_min, bbox_max, res)
                        mesh.export(sample_mesh_cano_path + '/{:03d}.ply'.format(i*batch_size+j))  
                        render_snapshot(mesh, sample_snapshot_cano_path + '/{:04d}.png'.format(i*batch_size+j))  
                        
                        encoding_expr = sample_latents[j:j+1, :, lat_dim_shape:]
                        condition = { 'geo': encoding_shape, 'exp': encoding_expr}
                        logits = get_logits( neural_3dmm, condition, grid_points.clone(), nbatch_points=25000)
                        deformed_mesh = mesh_from_logits(logits, bbox_min, bbox_max, res)
                        deformed_mesh.export(sample_mesh_expre_path + '/{:03d}.ply'.format(i*batch_size+j))  
                        print(deformed_mesh.vertices.shape, deformed_mesh.faces.shape)
                        render_snapshot(deformed_mesh, sample_snapshot_expre_path + '/{:04d}.png'.format(i*batch_size+j))    
                        