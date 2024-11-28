
import argparse
import torch
import json, os, yaml
import torch
import numpy as np
import os.path as osp

from omegaconf import OmegaConf
from dphm_tum import env_paths
from dphm_tum.data.face_dataset import ScannerJson
from dphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint
from dphm_tum.models.denoising_diffusion_pytorch_1d import Trainer1D, GaussianDiffusion1D, Unet1D
from dphm_tum.utils.load_checkpoint import load_pretrained_nphm_backward

parser = argparse.ArgumentParser(description='Run Model')
parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
parser.add_argument('-overfit', required=False, action='store_true')
parser.set_defaults(overfit=False)
try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))
config = OmegaConf.create(CFG)

exp_dir = env_paths.EXPERIMENT_DIR + '/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
if not os.path.exists(exp_dir):
    print('Creating checkpoint dir: ' + exp_dir)
    os.makedirs(exp_dir)
    with open(fname, 'w') as yaml_file:
        yaml.safe_dump(CFG, yaml_file, default_flow_style=False)
else:
    with open(fname, 'r') as f:
        print('Loading config file from: ' + fname)
        CFG = yaml.safe_load(f)
print(json.dumps(CFG, sort_keys=True, indent=4))


# load pretrained nphm backward model
neural_3dmm, latent_codes = load_pretrained_nphm_backward(config)

device = torch.device("cuda")
lm_inds = np.load( env_paths.ANCHOR_INDICES_PATH.format(config.nphm_backward.num_anchors) )
anchors = torch.from_numpy( np.load(env_paths.ANCHOR_MEAN_PATH.format(config.nphm_backward.num_anchors)) ).float().unsqueeze(0).unsqueeze(0).to(device)

# dataloader for latent diffusion
train_dataset = ScannerJson('train',
                                    CFG['training']['npoints_decoder'],
                                    CFG['training']['batch_size'],
                                    lm_inds=lm_inds,
                                    subject_json_path = config.nphm_backward.identity_json_path,
                                    expression_json_path = config.nphm_backward.expression_json_path,
                                    )
val_dataset = ScannerJson('val',
                                    CFG['training']['npoints_decoder'],
                                    CFG['training']['batch_size'],
                                    lm_inds=lm_inds,
                                    subject_json_path = config.nphm_backward.identity_json_path,
                                    expression_json_path = config.nphm_backward.expression_json_path,
                                    )
print('Lens of datasets right after creation')
print('Len of train dataset: {}'.format(len(train_dataset)))
print('Len of val dataset: {}'.format(len(val_dataset)))

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


trainer = Trainer1D(diffusion, train_dataset, val_dataset, 
                    _identity_embedding, None, None, 
                    _expression_embedding, None, None, 
                    neural3dmm = neural_3dmm,
                    overfit = args.overfit,
                    project = project,
                    exper_name = args.exp_name,
                    config=CFG,
                    train_batch_size = CFG['training']['batch_size'], 
                    val_batch_size = CFG['training']['batch_size_val'],  
                    train_lr = CFG['training']['lr'],               #8e-5,
                    train_num_steps = CFG['training']['num_steps'], #700000,        # total training steps
                    gradient_accumulate_every = CFG['training']['grad_acc'], #2,    # gradient accumulation steps
                    ema_update_every = CFG['training']['ema_update_every'],
                    ema_decay = CFG['training']['ema_decay'],          #0.995,      # exponential moving average decay
                    save_and_sample_every = CFG['training']['save_and_sample_every'],
                    num_samples = CFG['training']['num_samples'],      # 1
                    save_mesh = CFG['training']['save_mesh'],          # False,
                    diff_type = CFG['training']['diff_type'], 
                    deform_type = 'backward',
                    log_loss = True
                    )
trainer.train()
