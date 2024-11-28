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

from pytorch3d.transforms import  matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.ops import knn_points, knn_gather
from scipy.spatial import KDTree, cKDTree

from dphm_tum import env_paths
from dphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint
from dphm_tum.models.reconstruction import get_logits, get_vertex_color
from dphm_tum.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from dphm_tum.utils.load_checkpoint import load_pretrained_nphm_backward

parser = argparse.ArgumentParser(
    description='Run generation'
)
parser.add_argument('-weight_dir', default="pretrain_models/nphm_backward", type=str)
parser.add_argument('-iden', default=0, type=int)
parser.add_argument('-expr', default=0, type=int)
args = parser.parse_args()

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

# load model config files
weight_dir_shape = args.weight_dir 
fname_shape = weight_dir_shape + '/configs.yaml'
with open(fname_shape, 'r') as f:
    print('Loading config file from: ' + fname_shape)
    CFG = yaml.safe_load(f)
print('###########################################################################')
print('####################     Model Configs     #############################')
print('###########################################################################')
print(json.dumps(CFG, sort_keys=True, indent=4))

# load participant IDs that were used for training
fname_subject_index = f"{weight_dir_shape}/subject_train_index.json"
with open(fname_subject_index, 'r') as f:
    print('Loading subject index: ' + fname_subject_index)
    subject_index = json.load(f)

# load expression indices that were used for training
fname_subject_index = f"{weight_dir_shape}/expression_train_index.json"
with open(fname_subject_index, 'r') as f:
    print('Loading subject index: ' + fname_subject_index)
    expression_index = json.load(f)

# construct the NPHM models and latent codebook
device = torch.device("cuda")
modalities = ['geo', 'exp'] #, 'app']
n_lats = [len(subject_index), len(expression_index)] #, len(subject_index)]

neural_3dmm, latent_codes = construct_n3dmm(
    cfg=CFG,
    modalities=modalities,
    n_latents=n_lats,
    device=device,
    include_color_branch=False
    #include_color_branch=True
)

# load checkpoint from trained NPHM model, including the latent codes
ckpt = 6000
ckpt_path = osp.join(weight_dir_shape, 'checkpoints/checkpoint_epoch_{}.tar'.format(ckpt))
print('Loaded checkpoint from: {}'.format(ckpt_path))
load_checkpoint(ckpt_path, neural_3dmm, latent_codes)

reconstruction_cfg = {
'min': [-.55, -.5, -.95],
'max': [0.55, 0.75, 0.4],
'res': 256 #350  # small for faster reconstruction # use 256 or higher to grasp reconstructed geometry better
}
grid_points = create_grid_points_from_bounds(minimun=reconstruction_cfg['min'],
                                                maximum=reconstruction_cfg['max'],
                                                res=reconstruction_cfg['res'])
grid_points = torch.from_numpy(grid_points).cuda().float()
grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).cuda()

lat_rep_shape = latent_codes.codebook['geo'](torch.LongTensor([[args.iden]]).cuda())
lat_rep_exp = latent_codes.codebook['exp'](torch.LongTensor([[args.expr]]).cuda())
print(lat_rep_shape.shape, lat_rep_exp.shape)

condition = {'geo': lat_rep_shape, 'exp': lat_rep_exp}
logits = get_logits(neural_3dmm, condition, grid_points, nbatch_points=40000)
mesh = mesh_from_logits(logits.copy(), reconstruction_cfg['min'], reconstruction_cfg['max'],reconstruction_cfg['res'])

write_dir = './pretrain_sample_output'
os.makedirs(write_dir, exist_ok=True)

mesh.export(write_dir +'/' + f"id{args.iden}_ex{args.expr}.ply")
torch.cuda.empty_cache()
