import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

import os
import numpy as np
import wandb
import mcubes
import trimesh
import pyvista as pv
from glob import glob
from dphm_tum import env_paths 
from dphm_tum.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from dphm_tum.models.reconstruction import get_logits #, get_logits_forward, deform_mesh

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def render_snapshot(trim, path, vis_points=False, point_size=2.5, color='lightblue'):   
    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'
    #pv.global_theme.color = 'white'
    #pv.set_plot_theme('document')
     
    pl = pv.Plotter(off_screen=True)
    if vis_points:
        pl.add_points(trim, render_points_as_spheres=True, point_size=point_size, scalars=trim[:, 2])
    else:
        pl.add_mesh(trim, color=color)
        
    pl.reset_camera()
    pl.camera.position = (0, 0, 3)
    pl.camera.zoom(1.5)
    pl.set_viewup((0, 1, 0))
    pl.camera.view_plane_normal = (-0, -0, 1)
    #pl.show()
    pl.show(screenshot=path)
        

# small helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
            #print('self cond', x.shape)
        #print(x.shape)
        x = self.init_conv(x)
        #print(x.shape)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        
        out = self.final_conv(x)

        return out

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        minimum = -1.0,
        maximum = 1.0,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize
        self.minimum, self.maximum = minimum, maximum
        self.normalize = self.normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = self.unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False): 
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    
    def model_predictions_grad_correct(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, \
            grad_guide_fn=None, observations=None, decoder_shape=None, decoder_expre=None, cur_step=None, total_steps=None, cfg=dict()): 
        # add grad_guide_fn and cfg, inspired by single stage nerf in iccv 2023
        clip_denoised = cfg.get('clip_denoised', clip_x_start) #True)
        clip_range = cfg.get('clip_range', [-1, 1])
        guidance_gain = cfg.get('guidance_gain', 1.0)
        grad_through_unet = cfg.get('grad_through_unet', True)
        snr_weight_power = cfg.get('snr_weight_power', 0.5)
        
        if grad_guide_fn is not None and grad_through_unet:
            x.requires_grad = True
            grad_enabled_prev = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
        
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
            
        def correct(x_start, grad_enabled_prev):
            x_start = maybe_clip(x_start)
            
            if grad_through_unet:
                x_start_unnorm = self.unnormalize(x_start)
                loss = grad_guide_fn(x_start_unnorm, observations, decoder_shape, decoder_expre, cfg, cur_step, total_steps)
                grad = torch.autograd.grad(loss, x)[0]
            else:
                x_start.requires_grad = True
                x_start_unnorm = self.unnormalize(x_start)
                grad_enabled_prev = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                loss = grad_guide_fn(x_start_unnorm, observations, decoder_shape, decoder_expre, cfg, cur_step, total_steps)
                grad = torch.autograd.grad(loss, x_start)[0]
            torch.set_grad_enabled(grad_enabled_prev)
            x_start.detach_()
            
            sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            if False:
                x_start -= grad * (
                    (sqrt_one_minus_alpha_bar_t ** (2 - snr_weight_power * 2))
                    * (sqrt_alpha_bar_t ** (snr_weight_power * 2 - 1))
                    * guidance_gain)
            else:
                x_start -= grad * guidance_gain
            return x_start

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            # correct x_0 via the gradient of observation loss
            if grad_guide_fn is not None:
                x_start = correct(x_start, grad_enabled_prev)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
             # correct x_0 via the gradient of observation loss
            if grad_guide_fn is not None:
                x_start = correct(x_start, grad_enabled_prev)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
             # correct x_0 via the gradient of observation loss
            if grad_guide_fn is not None:
                x_start = correct(x_start, grad_enabled_prev)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img
    
    
    @torch.no_grad()
    def p_sample_loop_intermediate(self, shape, step=50):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        x_start = None

        result_list = []
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            
            if t % step == 0:
               result_list.append( self.unnormalize(img) ) 

        img = self.unnormalize(img)
        result_list.append(img)
        return result_list

    @torch.no_grad()
    def ddim_sample(self, shape, grad_guide_fn=None, observations=None, 
                                decoder_shape=None, decoder_expre=None,  cfg={}, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            if grad_guide_fn is not None:
                pred_noise, x_start, *_ = self.model_predictions_grad_correct(img, time_cond, self_cond, clip_x_start = clip_denoised, \
                    grad_guide_fn=grad_guide_fn, observations=observations, decoder_shape=decoder_shape, decoder_expre=decoder_expre, cfg=cfg, cur_step=total_timesteps-time, total_steps=total_timesteps)
            else:
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length))
    
    @torch.no_grad()
    def sample_intermediate(self, batch_size = 16, step=50):
        seq_length, channels = self.seq_length, self.channels
        return self.p_sample_loop_intermediate((batch_size, channels, seq_length), step=step)


    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

    def normalize_to_neg_one_to_one(self, x):
        x = (x - self.minimum) / (self.maximum - self.minimum)
        x = x * 2.0 - 1.0
        return x
    
    def unnormalize_to_zero_to_one(self, x):
        x = (x + 1) / 2
        x = x * (self.maximum - self.minimum) + self.minimum
        return x
    
    # 3d unconditional sds loss inspired by 2d sds loss of stable diffusion
    def forward_sds(self, img, t,  guidance_scale=100, grad_scale=1, skip_unet=False, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'

        img = self.normalize(img)
        return self.p_losses_sds(img, t, guidance_scale, grad_scale, skip_unet, *args, **kwargs)
    
    def p_losses_sds(self, x_start, t, guidance_scale, grad_scale, skip_unet, noise = None, clip_x_start=True):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            target = noise
            pred_noise = model_out
            x_start_new = self.predict_start_from_noise(x, t, pred_noise)
            x_start_new = maybe_clip(x_start_new)
            pred_noise = self.predict_noise_from_start(x, t, x_start_new)
        elif self.objective == 'pred_x0':
            target = x_start
            x_start_new = model_out
            x_start_new = maybe_clip(x_start_new)
            pred_noise = self.predict_noise_from_start(x, t, x_start_new)
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
            v = model_out
            x_start_new = self.predict_start_from_v(x, t, v)
            x_start_new = maybe_clip(x_start_new)
            pred_noise = self.predict_noise_from_start(x, t, x_start_new)

        if skip_unet:
            # w(t), sigma_t^2
            if False:
                sigma = (1 - self.alphas_cumprod[t]) 
                grad = (1.0 / torch.sqrt(sigma))[:, None, None] * (-pred_noise.detach())
                loss = grad * x_start
                loss = reduce(loss, 'b ... -> b (...)', 'mean')   
                #print('sds weight in diffusionNerf', (1.0 / torch.sqrt(sigma)))
            else:
                w = 1 - self.alphas_cumprod[t] #(1 - self.alphas[t])
                grad = grad_scale * w[:, None, None] * (pred_noise - noise) 
                grad = torch.nan_to_num(grad)
                targets = (x_start - grad).detach()
                loss = 0.5 * F.mse_loss(x_start.float(), targets, reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean')   
            return loss.mean()
        else:
            loss = F.mse_loss(model_out, target, reduction = 'none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')

            loss = loss * extract(self.loss_weight, t, loss.shape)
            return loss.mean()
# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        dataset_val: Dataset,
        latent_shape_codes: torch.Tensor,
        latent_shape_codes_val: torch.Tensor,
        decoder_shape: torch.nn.Module,
        latent_expre_codes: torch.Tensor,
        latent_expre_codes_val: torch.Tensor,
        decoder_expre: torch.nn.Module,
        *,
        config={},
        train_batch_size = 16,
        val_batch_size = 16,
        overfit=False,
        diff_type = "shape", #"expre", "joint",
        project = "shape_space_diffusion",
        exper_name = "npm_identity",
        deform_type = "forward", #"backward"
        neural3dmm = None,
        log_loss = True,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        save_mesh = False,
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
    ):
        super().__init__()
        
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        #assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.batch_size_val = val_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.overfit = overfit
        
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        dlv = DataLoader(dataset_val, batch_size = val_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())
        dlv = self.accelerator.prepare(dlv)
        self.dlv = cycle(dlv)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
            
        results_folder = env_paths.EXPERIMENT_DIR 
        self.results_folder = Path(results_folder)
        
        self.exper_folder = Path(results_folder + '/' + exper_name)
        self.exper_folder.mkdir(exist_ok = True)
        
        self.checkpoint_folder = Path(results_folder + '/' + exper_name + '/checkpoints') 
        self.checkpoint_folder.mkdir(exist_ok = True)
        
        self.sample_folder = Path( results_folder + '/' + exper_name + '/samples') 
        self.sample_folder.mkdir(exist_ok = True)
        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
        #
        self.latent_shape_codes = latent_shape_codes.to(self.device)
        self.latent_shape_codes_val = latent_shape_codes_val.to(self.device) if latent_shape_codes_val is not None else None
        if decoder_shape is not None:
            self.decoder_shape = decoder_shape.to(self.device) 
            self.decoder_shape.eval() 
        else:
            self.decoder_shape = None
        self.latent_expre_codes = latent_expre_codes.to(self.device)
        self.latent_expre_codes_val = latent_expre_codes_val.to(self.device) if latent_expre_codes_val is not None else None
        if decoder_expre is not None:
            self.decoder_expre = decoder_expre.to(self.device)
            self.decoder_expre.eval()
        else:
            self.decoder_expre = None
        if neural3dmm is not None:
            self.neural3dmm = neural3dmm.to(self.device)
            self.neural3dmm.eval()
        else:
            self.neural3dmm = None
        self.log_loss = log_loss
        #self.anchors = anchors
        self.min = [-.55, -.5, -.95] 
        self.max = [.55, 0.75, 0.4]  
        self.res = 256
        self.grid_points = create_grid_points_from_bounds(self.min, self.max, self.res)
        self.grid_points = torch.from_numpy(self.grid_points).to(self.device, dtype=torch.float)
        self.grid_points = torch.reshape(self.grid_points, (1, len(self.grid_points), 3)).to(self.device)
        
        if log_loss:
            #TODO
            wandb.login()
            wandb.init(project=project,  config=config, name=exper_name)
            wandb.watch(self.model, log_freq=100)
        self.cfg = config
        self.diff_type = diff_type
        self.deform_type = deform_type
        if deform_type == "forward":
            self.lat_dim_id = config['id_decoder']['decoder_lat_dim']
            self.lat_dim_expr = config['ex_decoder']['decoder_lat_dim']
        else:
            self.lat_dim_id = config['nphm_backward']['z_global_shape'] + (config['nphm_backward']['num_anchors'] + 1) * config['nphm_backward']['z_local_shape']
            self.lat_dim_expr = config['nphm_backward']['z_sdf']
        self.save_mesh = save_mesh

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            #'version': __version__
        }

        torch.save(data, str(self.checkpoint_folder / f'model-{milestone}.pt'))

    def load(self, milestone=None):
        accelerator = self.accelerator
        device = accelerator.device

        if milestone is not None:
            data = torch.load(str(self.checkpoint_folder / f'model-{milestone}.pt'), map_location=device)
        else:
            checkpoints = glob(str(self.checkpoint_folder) + '/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_folder))
                return 0

            checkpoints = [os.path.splitext(os.path.basename(path))[0][6:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            if 'ckpt' in self.cfg and self.cfg['ckpt'] is not None:
                path = str(self.checkpoint_folder) + '/'  + 'model-{}.pt'.format(self.cfg['ckpt'])
            else:
                print('LOADING', checkpoints[-1])
                path = str(self.checkpoint_folder) + '/'  + 'model-{}.pt'.format(checkpoints[-1])

            print('Loaded checkpoint from: {}'.format(path))
            data = torch.load(path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        return milestone

    def fetch_data(self):
        device = self.accelerator.device
        if self.deform_type == "forward":
            if self.overfit:
                idx = torch.zeros((self.batch_size, 1)).long().to(device)
                lat = self.latent_shape_codes(idx)
            else:
                data = next(self.dl) #.to(device)
                data_cuda = {k: v.to(device).float() for (k, v) in zip(data.keys(), data.values())}
                if self.diff_type == "shape":
                    idx = data_cuda.get('idx').long().to(device)
                    lat = self.latent_shape_codes(idx)
                elif self.diff_type == "expre":
                    idx = data_cuda.get('idx').long().to(device)
                    lat = self.latent_expre_codes(idx)
                else:
                    idx = data_cuda.get('idx').long().to(device)
                    expre_cond = self.latent_expre_codes(idx)
                    sub_ind = data_cuda.get('subj_ind').long().to(device)
                    shape_cond = self.latent_shape_codes(sub_ind)
                    lat = torch.cat([shape_cond, expre_cond], dim=-1).contiguous()
            return lat
        else:
            data = next(self.dl) #.to(device)
            data_cuda = {k: v.to(device).float() for (k, v) in zip(data.keys(), data.values())}
            idx = data_cuda.get('idx').long().to(device)
            expre_cond = self.latent_expre_codes(idx)
            sub_ind = data_cuda.get('subj_ind').long().to(device)
            shape_cond = self.latent_shape_codes(sub_ind)
            if self.diff_type == "shape":
                lat = shape_cond
            elif self.diff_type == "expre":
                lat = expre_cond
            else:
                lat = torch.cat([shape_cond, expre_cond], dim=-1).contiguous()
            return lat, shape_cond, expre_cond

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        milestone = self.load()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if self.deform_type == "forward":
                        lat = self.fetch_data()
                    else:
                        lat, shape_cond, expre_cond = self.fetch_data()

                    with self.accelerator.autocast():
                        loss = self.model(lat) 
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                
                #
                if self.log_loss:
                    wandb.log({'loss': total_loss})

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                        all_samples = torch.cat(all_samples_list, dim = 0)
                        
                        if self.save_mesh:
                            self.sample_folder_milestone = Path( str(self.sample_folder) + '/{:d}'.format(milestone) )
                            self.sample_folder_milestone.mkdir(exist_ok = True)
                        self.sample_folder_milestone_snapshot = Path( str(self.sample_folder) + '/{:d}_snapshot'.format(milestone) )
                        self.sample_folder_milestone_snapshot.mkdir(exist_ok = True)
                        cano_mesh, cano_anchors = None, None
                        for i in range(all_samples.shape[0]):
                            if self.deform_type == 'forward':
                                if self.diff_type == "shape":
                                    sampled_mesh = self.construct_rec(all_samples[i:i+1])
                                elif self.diff_type == "expre":
                                    shape_idx = torch.ones((1, 1)).long().to(device) * (milestone % self.latent_shape_codes.weight.data.shape[0])
                                    cano_mesh, sampled_mesh, cano_anchors = self.construct_re_and_deform(self.latent_shape_codes(shape_idx), all_samples[i:i+1], cano_mesh, cano_anchors)
                                else:
                                    _, sampled_mesh, _ = self.construct_re_and_deform(all_samples[i:i+1, :, :self.lat_dim_id], all_samples[i:i+1, :, self.lat_dim_id:], None)
                            else:
                                if self.diff_type == "shape":
                                    sampled_mesh = self.construct_rec_backward(all_samples[i:i+1], torch.zeros(expre_cond[0:1].shape).float().to(self.device))
                                elif self.diff_type == "expre":
                                    sampled_mesh = self.construct_rec_backward(shape_cond[0:1], all_samples[i:i+1])
                                else:
                                    sampled_mesh  = self.construct_rec_backward(all_samples[i:i+1, :, :self.lat_dim_id], all_samples[i:i+1, :, self.lat_dim_id:])
                            if self.save_mesh:
                                sampled_mesh.export( str(self.sample_folder_milestone) + '/{:d}.ply'.format(i) )
                            render_snapshot( sampled_mesh, str(self.sample_folder_milestone_snapshot) + '/{:d}.png'.format(i) )
                        self.save(milestone)

                pbar.update(1)
        accelerator.print('training complete')


    def construct_rec(self, encoding, return_anchors=False):
        if return_anchors:
            logits, anchors = get_logits_forward(self.decoder_shape,
                                encoding,
                                self.grid_points.clone(),
                                nbatch_points=25000,
                                return_anchors=True
                    )
        else:
            logits = get_logits_forward(self.decoder_shape,
                                encoding,
                                self.grid_points.clone(),
                                nbatch_points=25000,
                    )
        trim = mesh_from_logits(logits, self.min, self.max, self.res)
        
        if return_anchors:
            return trim, anchors
        else:
            return trim
        
    def construct_re_and_deform(self, encoding_shape, encoding_expr, mesh, anchors=None):

        if mesh is None:
            # reconstruct neutral geometry from implicit repr.
            logits, anchors = get_logits_forward(decoder=self.decoder_shape,
                                encoding=encoding_shape,
                                grid_points=self.grid_points.clone(),
                                nbatch_points=25000,
                                return_anchors=True)
            mesh = mesh_from_logits(logits, self.min, self.max, self.res)
        deformed_mesh = deform_mesh(mesh, self.decoder_expre, encoding_expr, anchors, lat_rep_shape=encoding_shape)
        return mesh, deformed_mesh, anchors
    
    def construct_rec_backward(self, encoding_shape, encoding_expr):
        condition = {'geo': encoding_shape, 'exp': encoding_expr}
        logits = get_logits( self.neural3dmm, condition, grid_points=self.grid_points.clone(), nbatch_points=25000)
        mesh = mesh_from_logits(logits, self.min, self.max, self.res)
        return mesh
        