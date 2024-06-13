import torch
from torch import nn
from typing import Optional, Literal, List
from inspect import getfullargspec

from dphm_tum.models.diff_operators import gradient
from dphm_tum.models.canonical_space import get_id_model
from dphm_tum.models.deformations import get_ex_model



class nn3dmm(nn.Module):
    '''
    Implemntation of a neural parametric head model, that encapsulates an id_model and ex_model.
    The id_model represents an SDF in canonical space-
    The ex_model represents a backward deformation field.
    (Forward deformations, as well, as no-deformations (monolith) are currently not supported.)
    '''
    def __init__(self,
                 id_model,
                 ex_model,
                 expr_direction : Optional[Literal['forward', 'backward', 'monolith']],
                 neutral_only : bool = False):
        super().__init__()

        self.id_model = id_model
        self.ex_model = ex_model
        self.expr_direction = expr_direction
        self.neutral_only =neutral_only


    def forward(self,
                in_dict,
                cond,
                return_grad : bool = False,
                return_can : bool = False,
                skip_color : bool = False,
                ignore_deformations : bool = False,
                ):
        '''
        Executre a forward pass of the parametric head model, including a call to the ex_model and id_model.
        To be specific, quries in posed space are backward warped into canonical space.
        There the SDF (and texture field) are executed to return the desired field values.
        :param in_dict: holds 3D query positions in posed space
        :param cond: A dictionary of latent codes, namely a code "geo" for the geometry to condition the SDF, as well as,
                    as a code "exp" describing the expression
        :param return_grad: if True, the gradient of the SDF w.r.t. the queries in posed space is included in the return dictionary
        :param return_can: if True, the queries in canonical space are included in the return dictionary
        :param skip_color: if True, the texture field (if it exists) is not queried
        :param ignore_deformations: should always be False
        :return: Returns a dictionary with a field "sdf" that holds the predicted SDF values.
                If the id_model also contains a texture field,
                 the return dictionary includes a field "color" with predicted RGB colors.
                Boolean flags determine whether spatial gradients and canonical coordinates should be returned.
        '''

        if return_grad:
            in_dict['queries'].requires_grad_()

        in_dict.update({'cond': cond})

        if self.expr_direction == 'forward':
            raise ValueError("Forward deformations are not currently implemented!!!")

        elif self.expr_direction == 'backward':
            # provide expression network with predicted facial anchors
            if hasattr(self.id_model, 'mlp_pos') and self.id_model.mlp_pos is not None and 'anchors' not in in_dict:
                in_dict.update({'anchors': self.id_model.get_anchors(cond['geo'])})

            # perform backward deformation
            if not self.neutral_only:

                # not sure where this would ever be used
                if ignore_deformations:
                    out_ex = self.ex_model(in_dict)
                    queries_canonical = in_dict['queries']
                    if self.ex_model.n_hyper > 0:
                        queries_canonical = torch.cat([queries_canonical, torch.zeros_like(out_ex['hyper_coords'])], dim=-1)
                else:
                    out_ex = self.ex_model(in_dict)
                    queries_canonical = in_dict['queries'] + out_ex['offsets']
                    # append predicted hyper dimensions
                    if self.ex_model.n_hyper > 0:
                        queries_canonical = torch.cat([queries_canonical, out_ex['hyper_coords']], dim=-1)
            # do nothing
            else:
                out_ex = {'offsets': torch.zeros_like(in_dict['queries'])}
                queries_canonical = in_dict['queries']

            in_dict.update({'queries_can': queries_canonical, 'offsets': out_ex['offsets']})

            # query id_model
            if skip_color in getfullargspec(self.id_model.forward)[0]:
                pred = self.id_model(in_dict, skip_color=skip_color)
            else:
                pred = self.id_model(in_dict)

            # mechanism propsed in the ImFace paper, not sure if this option is still supported properly
            if self.ex_model.sdf_corrective:
                pred['sdf'] += out_ex['sdf_corrective']
                pred['sdf_corrective'] = out_ex['sdf_corrective']

            # return predicted hyper dimenions such that regularization can be applied later
            if self.ex_model.n_hyper > 0:
                pred['hyper_coords'] = out_ex['hyper_coords']

            if return_grad:
                grad = gradient(pred['sdf'], in_dict['queries'])
                pred.update({'gradient': grad})

            # return predicted deformations such that regularization can be applied later
            pred.update({'offsets': out_ex['offsets']})

            if return_can:
                pred.update({'queries_can': queries_canonical})

            return pred

        elif self.expr_direction == 'monolith':
            raise ValueError('Monolithic model formatulation (id_model and ex_model represented by a single NN, as in SSIF) is not suuported anymore')
            return self.ex_model(queries, lat_ex, lat_id)

        else:
            raise ValueError(f'unexpected value for {self.expr_direction}')


class LatentCodes(nn.Module):
    '''
    Simple wrapper class that encapsulates different modalities of latent codes into one.
    I.e. there can be latent codes for identity geometry ("geo") and appearance ("app"), as well as, for expressions ("exp").
    '''

    def __init__(self,
                 n_latents : List[int],
                 n_channels : List[int],
                 modalities : List[Literal['geo', 'app', 'exp']],
                 types : List[Literal['vector', 'grid', 'triplane']],
                 resolutions : Optional[List[Optional[int]]] = None,
                 init_zeros : bool = False,
                 variational : bool = False,
                 ):
        '''
        Instatiate the LatenCodes.
        :param n_latents: Number of latent codes, one for each modlaity
        :param n_channels: Number of latent dimensions for each modality
        :param modalities: Names of the modalities, e.g. ["geo", "exp"] or ["geo", "app", "exp"]
        :param types: Types for each latent code, currenly only "vector" is supported
        :param resolutions: Kind of deprecated, only relevant for "grid"-type latent codes
        :param init_zeros: Whether latent codes should be initilized with zeros
        :param variational: Kind of deprecated. Set to True for variational auto-decoder training, but that might not be supported anymore.
        '''
        super(LatentCodes, self).__init__()

        self.variational = variational

        self.codebook = torch.nn.ModuleDict()

        # Initialize the individual latent code for all modalities
        for i, mod in enumerate(modalities):
            self.codebook[mod] = SingleModalityLatentCodes(n_latents[i],
                                                           n_channels[i],
                                                           resolutions[i] if resolutions is not None else None,
                                                           types[i],
                                                           init_zeros=init_zeros,
                                                           )

        # In the variational case, also the variances have to be initialized.
        if self.variational:
            self.codebook_logvar = torch.nn.ModuleDict()

            for i, mod in enumerate(modalities):
                self.codebook_logvar[mod] = SingleModalityLatentCodes(n_latents[i],
                                                                       n_channels[i],
                                                                       resolutions[i] if resolutions is not None else None,
                                                                       types[i],
                                                                       init_zeros=init_zeros,
                                                                       )

    def forward(self, latent_idx, return_mu_sig=False):
        '''
        :param latent_idx: dictionary of LongTensors, having one index tensor per modality
        :param return_mu_sig: if True, and the self.variational==True mu and sigma are returned next to the sampled codes.
        :return: Returns a dictionary of Tensors, one for each modality
        '''

        # Not used
        if self.variational:
            code_dict  = {}
            mu_dict = {}
            log_var_dict = {}
            for mod in self.codebook.keys():
                log_var = self.codebook_logvar[mod](latent_idx[mod])
                std = torch.exp(0.5 * log_var) #+ 1e-8
                eps = torch.randn_like(std)
                mu = self.codebook[mod](latent_idx[mod])
                code_dict[mod] = eps * std + mu
                mu_dict[mod] = mu
                log_var_dict[mod] = log_var
            if return_mu_sig:
                return code_dict, mu_dict, log_var_dict
            else:
                return code_dict
        # simply index each embedding layer and place the results in a dict
        else:
            code_dict =  {mod: codes(latent_idx[mod]) for (mod, codes) in self.codebook.items()}
        return code_dict


class SingleModalityLatentCodes(nn.Module):
    '''
    Wrapper Class for different types of latent codes.
    The idea was to encapsulate the use of vectors, grids and triplanes.
    Currently, only vectors are supported.
    In the vector case, this is simply an indexing in the embedding layer.
    '''
    def __init__(self,
                 n_latents : int,
                 n_channels : int,
                 resolution : Optional[int] = None,
                 type : Literal['vector', 'grid', 'triplane'] = 'vector',
                 init_zeros : bool = False):
        super(SingleModalityLatentCodes, self).__init__()

        if type in ['grid', 'triplane']:
            assert resolution is not None and resolution > 0
        else:
            assert resolution is None

        if type == 'vector':
            dim = n_channels
        elif type == 'grid':
            dim = resolution ** 3
        elif type == 'triplane':
            dim = 3*resolution**2
        else:
            raise ValueError(f'Unexpected value for latent type encountered: {type}!')

        self.embedding = torch.nn.Embedding(n_latents, dim,
                           max_norm=1.0, sparse=True, device='cuda').float()

        if init_zeros:
            torch.nn.init.zeros_(
                self.embedding.weight.data,
            )
        else:
            torch.nn.init.normal_(
                self.embedding.weight.data,
                0.0,
                0.001,
            )

    def forward(self, input_idx):
        return self.embedding(input_idx)



def construct_n3dmm(cfg : dict,
                    modalities : List[Literal['geo', 'app', 'exp']],
                    n_latents : list[int],
                    device : torch._C.device = 0,
                    include_color_branch : bool = False
                    ):
    '''
    Construct a neural parametric head model from a given config dictionary.


    :param cfg: Dictionary holding all neccessary configs
    :param modalities: Modalities that shall be included
    :param n_latents: Number of latent codes per modality
    :param device: Torch Device
    :return: nn3dmm instance, LatentCodes instance
    '''

    # hack needed for legacy exoperiments
    if 'n_hyper' not in cfg['decoder']:
        cfg['decoder']['n_hyper'] = 2
        cfg['decoder']['lambda_hyper'] = 0.01
    id_model, anchors = get_id_model(cfg['decoder'],
                                     spatial_input_dim=3+cfg['decoder']['n_hyper'],
                                     rank=device,
                                     include_color_branch=include_color_branch)

    ex_model = get_ex_model(cfg, anchors)
    n3dmm = nn3dmm(id_model=id_model,
                   ex_model=ex_model,
                   expr_direction='backward',
                   ).to(device)

    #latent_codes = LatentCodes(n_latents=n_latents,
    #                           n_channels=[id_model.lat_dim, ex_model.lat_dim_expr, id_model.lat_dim,],
    #                           modalities=modalities,
    #                           types=['vector']*len(modalities),
    #                           ).to(device)
    latent_codes = LatentCodes(n_latents=n_latents,
                               n_channels=[id_model.lat_dim_glob + id_model.lat_dim_loc_geo * (id_model.n_anchors + 1),
                                           ex_model.lat_dim_expr,
                                           id_model.lat_dim_glob + id_model.lat_dim_loc_app * (id_model.n_anchors + 1), ],
                               modalities=modalities,
                               types=['vector'] * len(modalities),
                               ).to(device)
    return n3dmm, latent_codes

def load_checkpoint(ckpt_path : str,
                    n3dmm : nn3dmm,
                    latent_codes : LatentCodes,
                    device='cuda',
                    ):
    '''
    Loads a training checkpoint into the neural 3dmm and latent codes instances
    '''
    print('Loaded checkpoint from: {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location=device)

    n3dmm.id_model.load_state_dict(checkpoint['id_decoder'], strict=True)
    n3dmm.ex_model.load_state_dict(checkpoint['ex_decoder'], strict=True)

    latent_codes.load_state_dict(checkpoint['latent_codes'])