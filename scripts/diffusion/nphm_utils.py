import json, os, yaml
import os.path as osp
import torch
from dphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint


def load_pretrained_nphm_backward(config):
    # load nphm pretrained weights, config, train split
    nphm_weight_dir = config.nphm_backward.nphm_weight_dir # #"/cluster/balrog/jtang/Head_tracking/NPHM-TUM/pretrain_nphm"
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