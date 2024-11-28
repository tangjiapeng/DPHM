import json, os, yaml
import os.path as osp
import torch
import numpy as np
from glob import glob
from dphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint
from dphm_tum.models.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Unet1D

def load_pretrained_nphm_backward(config):
    # load nphm pretrained weights, config, train split
    nphm_weight_dir = config.nphm_backward.nphm_weight_dir 
    fname_config = nphm_weight_dir + '/configs.yaml'
    with open(fname_config, 'r') as f:
        print('Loading config file from: ' + fname_config)
        CFG_NPHM = yaml.safe_load(f)
    
    # load participant IDs that were used for training
    fname_subject_index = f"{nphm_weight_dir}/subject_train_index.json"
    with open(fname_subject_index, 'r') as f:
        print('Loading subject index: ' + fname_subject_index)
        subject_index = json.load(f)

    # load expression indices that were used for training
    fname_subject_index = f"{nphm_weight_dir}/expression_train_index.json"
    with open(fname_subject_index, 'r') as f:
        print('Loading subject index: ' + fname_subject_index)
        expression_index = json.load(f)

    # construct the NPHM models and latent codebook
    device = torch.device("cuda")
    modalities = ['geo', 'exp'] #, 'app']
    n_lats = [len(subject_index), len(expression_index)] #, len(subject_index)]

    neural_3dmm, latent_codes = construct_n3dmm(
        cfg=CFG_NPHM,
        modalities=modalities,
        n_latents=n_lats,
        device=device,
        include_color_branch=False
        #include_color_branch=True
    )

    # load checkpoint from trained NPHM model, including the latent codes
    ckpt_path = config.nphm_backward.pretrained_model
    # ckpt = config.nphm_backward.ckpt #6000
    # ckpt_path = osp.join(nphm_weight_dir, 'checkpoints/checkpoint_epoch_{}.tar'.format(ckpt))
    # print('Loaded checkpoint from: {}'.format(ckpt_path))
    load_checkpoint(ckpt_path, neural_3dmm, latent_codes)
    
    neural_3dmm.eval() 
    latent_codes.eval()
    return neural_3dmm, latent_codes


def load_pretrained_diffusion_model(diffusion, checkpoint_dir, device):
    checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
    checkpoints = glob(checkpoint_dir+'/*')
    checkpoints = [os.path.splitext(os.path.basename(path))[0][6:] for path in checkpoints]
    checkpoints = np.array(checkpoints, dtype=int)
    checkpoints = np.sort(checkpoints)
    ckpt = checkpoints[-1]
    path = osp.join(checkpoint_dir, 'model-{}.pt'.format(ckpt) )
    print('Loaded checkpoint from: {}'.format(path))
    data = torch.load(path, map_location=device)
    diffusion.load_state_dict(data['model'])
    return diffusion


def load_pretrained_identity_and_expression_diffusion(config, device, latent_codes):
    _identity_latent_codes = latent_codes.codebook['geo'].embedding.weight.data
    _expression_latent_codes = latent_codes.codebook['exp'].embedding.weight.data
    print("identity latent code size of training set :", _identity_latent_codes.shape)
    print("expression latent code size of training set :", _expression_latent_codes.shape)
    _identity_latent_codes_min, _identity_latent_codes_max = _identity_latent_codes.min(), _identity_latent_codes.max()
    _expression_latent_codes_min, _expression_latent_codes_max = _expression_latent_codes.min(), _expression_latent_codes.max()
    print('training identity latent code min max :', _identity_latent_codes_min, _identity_latent_codes_max)
    print('training expression latent code min max :', _expression_latent_codes_min, _expression_latent_codes_max)
    
    # load shape diffu config
    with open(config.diffusion.cfg_file_shape_diff, 'r') as f:
        print('Loading config file from: ' + config.diffusion.cfg_file_shape_diff)
        CFG_SHAPE_DIFF = yaml.safe_load(f)
    # define shape denoiser
    model_shape = Unet1D(
        **CFG_SHAPE_DIFF['diffusion']['net_kwargs'],
    )
    model_shape = model_shape.to(device)
    # define shape diffusion
    diffusion_shape = GaussianDiffusion1D(
        model_shape,
        **CFG_SHAPE_DIFF['diffusion']['diff_kwargs'],
        minimum = _identity_latent_codes_min,
        maximum = _identity_latent_codes_max,
    )
    diffusion_shape = diffusion_shape.to(device)
    diffusion_shape = load_pretrained_diffusion_model(diffusion_shape, config.diffusion.exp_dir_shape_diff, device)

    # load expre diff config
    with open(config.diffusion.cfg_file_expre_diff, 'r') as f:
        print('Loading config file from: ' + config.diffusion.cfg_file_expre_diff)
        CFG_EXPRE_DIFF = yaml.safe_load(f)
    # define expre denoiser
    model_expre = Unet1D(
        **CFG_EXPRE_DIFF['diffusion']['net_kwargs'],
    )
    model_expre = model_expre.to(device)
    # define expre diffusion 
    diffusion_expre = GaussianDiffusion1D(
        model_expre,
        **CFG_EXPRE_DIFF['diffusion']['diff_kwargs'],
        minimum = _expression_latent_codes_min,
        maximum = _expression_latent_codes_max,
    )
    diffusion_expre = diffusion_expre.to(device)
    diffusion_expre = load_pretrained_diffusion_model(diffusion_expre, config.diffusion.exp_dir_expre_diff, device)
    
    return diffusion_shape, diffusion_expre
