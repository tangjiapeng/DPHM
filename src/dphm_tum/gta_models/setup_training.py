import numpy as np
import torch

# data
from dphm_tum import env_paths
from dphm_tum.data.face_dataset import NPHMdataset

# NNs
from dphm_tum.models.canonical_space import get_id_model
from dphm_tum.models.deformations import ZeroDeformation, DeformationNetwork
from dphm_tum.models.neural3dmm import LatentCodes






def set_up_datasets(cfg,
                     \
                     lm_inds = None,
                     debug_run : bool = False,
                     neutral_only : bool = False,
                     no_validation : bool = False,
                     use_patches : bool = False,
                     uv_communication : bool = False,
                    **kwards,
                    ):
    DataSetClass = NPHMdataset

    if debug_run:
        cfg['training']['batch_size'] = 10
        cfg['training']['npoints_face'] = 100
        cfg['training']['npatches_per_batch'] = 2
        cfg['training']['ckpt_interval'] = 100

    train_dataset = DataSetClass(
                                 mode='train',
                                      n_supervision_points_corresp=cfg['training'].get('npoints_corresp', 250),
                                      n_supervision_points_face=cfg['training'].get('npoints_face', 1000),
                                      n_supervision_points_non_face=cfg['training'].get('npoints_non_face', 250),
                                      n_supervision_points_off_surface=cfg['training'].get('npoints_off_surface', 250),
                                      batch_size=cfg['training']['batch_size'],
                                      # sigma_near=CFG['training']['sigma_near'], #TODO probably should use that?!
                                      lm_inds=lm_inds,
                                      # is_closed=args.closed
                                      neutral_only=neutral_only,
                                      no_validation=no_validation,
                                      num_anchors=cfg['decoder']['id'].get('nloc'), #, 39),
                                      num_symm=cfg['decoder']['id'].get('nsymm_pairs'), #, 16),
                                      )
    val_dataset = DataSetClass(mode='val',
                                   n_supervision_points_corresp=cfg['training'].get('npoints_corresp', 250),
                                   n_supervision_points_face=cfg['training'].get('npoints_face', 1000),
                                   n_supervision_points_non_face=cfg['training'].get('npoints_non_face', 250),
                                   n_supervision_points_off_surface=cfg['training'].get('npoints_off_surface', 250),
                                    batch_size=cfg['training']['batch_size'],
                                    # sigma_near=CFG['training']['sigma_near'], #TODO probably should use that?!
                                    lm_inds=lm_inds,
                                    # is_closed=args.closed
                                    neutral_only=neutral_only,
                                    no_validation=no_validation,
                                    num_anchors=cfg['decoder'].get('decoder_nloc', 39),
                                   num_symm=cfg['decoder'].get('nsymm_pairs', 16),
                               )

    return train_dataset, val_dataset


def set_up_networks(cfg,
                    model_type,
                    rank=None,
                    \
                    anchors = None,
                    neutral_only : bool = False,
                    n_hyper : int = 0,
                    include_app : bool = False,
                    pass_exp2app : bool = False,
                    old_global : bool = False,
                    legacy_nphm: bool = False,
                    omit_extra_mlp : bool = False,
                    modulation_in_communcation : bool = False,
                    ignore_outer_mlp : bool = True,
                    pass_pos : bool = False,
                    is_monolith : bool = False,
                    uv_communication : bool = False,
                    intermediate_nphm : bool = False,
                    old_old = False,
                    global_color : bool = False,
                    disable_color_communication : bool = False,
                    **kwargs,
                    ):

    if model_type == 'nphm':

        id_decoder, _ = get_id_model(cfg['decoder'],
                                  3 + n_hyper,
                                  include_color_branch=include_app,
                                  rank=rank,
                                  )


        if neutral_only:
            ex_decoder = ZeroDeformation()
        else:
            ex_decoder = DeformationNetwork(mode=cfg['decoder']['ex']['mode'],
                                            lat_dim_expr=cfg['decoder']['ex']['lat_dim_ex'],
                                            lat_dim_id=cfg['decoder']['ex']['lat_dim_id'],
                                            lat_dim_glob_shape=cfg['decoder']['id']['lat_dim_glob'],
                                            lat_dim_loc_shape=cfg['decoder']['id']['lat_dim_loc_geo'],
                                            n_loc=cfg['decoder']['id']['nloc'],
                                            anchors=anchors,
                                            hidden_dim=cfg['decoder']['ex']['hidden_dim'],
                                            nlayers=cfg['decoder']['ex']['nlayers'],
                                            out_dim=3,
                                            input_dim=3,
                                            neutral_only=neutral_only or is_monolith,
                                            n_hyper=n_hyper,
                                            sdf_corrective=False,  # TODO lambda_sdf_corrective > 0,
                                            local_arch=False,  # TODOlocal_def_arch,
                                            )
            #if is_monolith:
            #   cfg['decoder']['ex']['lat_dim_ex'] = id_decoder.lat_dim
            #   ex_decoder.lat_dim_expr = id_decoder.lat_dim
            #if local_def_arch:
            #   CFG['decoder']['ex']['lat_dim_ex'] = id_decoder.lat_dim
            #   ex_decoder.lat_dim_expr = id_decoder.lat_dim

    elif model_type == 'global':
        if old_global:
            assert 1 == 2
            print('OLDGLOBALFIELD')
            print('OLDGLOBALFIELD')
            print('OLDGLOBALFIELD')
            id_decoder = GlobalField(
                lat_dim=cfg['decoder']['id']['lat_dim'] if not is_monolith else cfg['decoder']['id']['lat_dim'] + cfg['decoder']['ex']['lat_dim_ex'],
                lat_dim_app=cfg['decoder']['id']['lat_dim_app'],
                hidden_dim=cfg['decoder']['id']['hidden_dim'],
                nlayers=cfg['decoder']['id']['nlayers'],
                geometric_init=True,
                out_dim=1,
                input_dim=3 + n_hyper,
                color_branch=include_app,
                num_freq_bands=0, #8,
                freq_exp_base=0.5,
                lat_dim_exp=cfg['decoder']['ex']['lat_dim_ex'] if pass_exp2app else 0,
                is_monolith=is_monolith,
            )
        else:
            print('NEWGLOBALFIELD')
            print('NEWGLOBALFIELD')
            print('NEWGLOBALFIELD')
            id_decoder = GlobalFieldNew(
            lat_dim=cfg['decoder']['id']['lat_dim'] if not is_monolith else cfg['decoder']['id']['lat_dim'] + cfg['decoder']['ex']['lat_dim_ex'],
            lat_dim_app=cfg['decoder']['id']['lat_dim_app'],
            hidden_dim=cfg['decoder']['id']['hidden_dim'],
            nlayers=cfg['decoder']['id']['nlayers'],
            nlayers_color=cfg['decoder']['id'].get('nlayers_color', 6),
            out_dim=1,
            input_dim=3 + n_hyper,
            color_branch=include_app,
            num_freq_bands=cfg['decoder']['id'].get('nfreq_bands_geo', 0),
            freq_exp_base=cfg['decoder']['id'].get('freq_base_geo', 0.5),
            lat_dim_exp=cfg['decoder']['ex']['lat_dim_ex'] if pass_exp2app else 0,
            num_freq_bands_color=cfg['decoder']['id'].get('nfreq_bands_color', 0),
            freq_exp_base_color=cfg['decoder']['id'].get('freq_base_color', 2.0),
            is_monolith=is_monolith,
            communication_dim=0, #16, #cfg['decoder']['id'].get('communication_dim', 0),
            uv_communication=uv_communication,
                include_anchors=True, #TODO make it an option, not the default
                anchors=anchors,
        )

        if neutral_only:
            ex_decoder = ZeroDeformation()
        else:
            ex_decoder = DeformationNetwork(mode=cfg['decoder']['ex']['mode'],
                                            lat_dim_expr=cfg['decoder']['ex']['lat_dim_ex'],
                                            lat_dim_id=-1,
                                            lat_dim_glob_shape=cfg['decoder']['id']['lat_dim'],
                                            lat_dim_loc_shape=-1,
                                            n_loc=-1,  # CFG['decoder']['id']['nloc'],
                                            anchors=None,
                                            hidden_dim=cfg['decoder']['ex']['hidden_dim'],
                                            nlayers=cfg['decoder']['ex']['nlayers'],
                                            out_dim=3,
                                            input_dim=3,
                                            neutral_only=neutral_only or is_monolith,
                                            n_hyper=n_hyper,
                                            sdf_corrective=False,  # TODO lambda_sdf_corrective > 0,
                                            local_arch=False,  # TODOlocal_def_arch,
                                            )

    elif model_type == 'grid':
        id_decoder = LatentGrid(channels_geo=cfg['decoder']['id']['lat_dim_geo'],
                                channels_app=cfg['decoder']['id']['lat_dim_app'],
                                resolution=cfg['decoder']['id']['grid_res'],
                                n_layers_geo=cfg['decoder']['id']['nlayers_geo'],
                                n_layers_app=cfg['decoder']['id']['nlayers_app'],
                                hidden_dim_geo=cfg['decoder']['id']['hidden_dim_geo'],
                                hidden_dim_app=cfg['decoder']['id']['hidden_dim_app'],
                                color_branch=include_app,
                                communcation_dim=16,
                                )
        if neutral_only:
            ex_decoder = ZeroDeformation()
        else:
            raise ValueError('Not implemented yet: latent grid + deformations')

    elif model_type == 'triplane':
        id_decoder = LatentTriPlane(
            channels_geo=cfg['decoder']['id']['lat_dim_geo'],
            channels_app=cfg['decoder']['id']['lat_dim_app'],
            resolution=cfg['decoder']['id']['grid_res'],
            n_layers_geo=cfg['decoder']['id']['nlayers_geo'],
            n_layers_app=cfg['decoder']['id']['nlayers_app'],
            hidden_dim_geo=cfg['decoder']['id']['hidden_dim_geo'],
            hidden_dim_app=cfg['decoder']['id']['hidden_dim_app'],
            color_branch=include_app,
            communcation_dim=16,
        )
        if neutral_only:
            ex_decoder = ZeroDeformation()
        else:
            raise ValueError('Not implemented yet: latent triplane + deformations')

    elif model_type == 'eg3d':
        id_decoder = EG3D(
            lat_dim_geo=cfg['decoder']['id']['lat_dim_geo'],
            lat_dim_app=cfg['decoder']['id']['lat_dim_app'],
            channels_geo=32,  # CFG['decoder']['id']['lat_dim_geo'],
            channels_app=32,  # CFG['decoder']['id']['lat_dim_app'],
            resolution=cfg['decoder']['id']['grid_res'],
            n_layers_geo=cfg['decoder']['id']['nlayers_geo'],
            n_layers_app=cfg['decoder']['id']['nlayers_app'],
            hidden_dim_geo=cfg['decoder']['id']['hidden_dim_geo'],
            hidden_dim_app=cfg['decoder']['id']['hidden_dim_app'],
            color_branch=include_app,
            communcation_dim=16,
        )
        print(id_decoder)  # , (1, CFG['decoder']['id']['lat_dim_app']))
        if neutral_only:
            ex_decoder = ZeroDeformation()
        else:
            raise ValueError('Not implemented yet: latent triplane + deformations')

    elif model_type == 'latent-mesh':
        assert 1 == 2, 'Double Check!'
        if mvs_dataset:
            anchors_overfit = np.load(f'{env_paths.ASSETS}/id{overfit_id:03d}_anchors_flame_up.npy')
            triangle_indices = np.load(f'{env_paths.ASSETS}/id{overfit_id:03d}_indices_flame_up.npy')
            triangle_normals = np.load(f'{env_paths.ASSETS}/id{overfit_id:03d}_normals_flame_up.npy')
        else:
            anchors_overfit = np.load(f'{env_paths.ASSETS}/id{overfit_id:03d}_anchors_flame_up_nphm.npy')
            triangle_indices = np.load(f'{env_paths.ASSETS}/id{overfit_id:03d}_indices_flame_up_nphm.npy')
            triangle_normals = np.load(f'{env_paths.ASSETS}/id{overfit_id:03d}_normals_flame_up_nphm.npy')
        flame_uvs = np.load(f'{env_paths.ASSETS}/flame_uvs.npy')
        flame_uv_vert_inds = np.load(f'{env_paths.ASSETS}/flame_uvs_vert_inds.npy')
        flame_uv_faces = np.load(f'{env_paths.ASSETS}/flame_uv_faces.npy')
        flame_uv_face_normals = np.load(f'{env_paths.ASSETS}/flame_uv_face_normals.npy')

        id_decoder = LatentPointCloud(channels_geo=CFG['decoder']['id']['lat_dim_geo'],
                                      channels_app=CFG['decoder']['id']['lat_dim_app'],
                                      anchors=torch.from_numpy(anchors_overfit),
                                      resolution=CFG['decoder']['id']['grid_res'],
                                      n_layers_geo=CFG['decoder']['id']['nlayers_geo'],
                                      n_layers_app=CFG['decoder']['id']['nlayers_app'],
                                      hidden_dim_geo=CFG['decoder']['id']['hidden_dim_geo'],
                                      hidden_dim_app=CFG['decoder']['id']['hidden_dim_app'],
                                      color_branch=color_branch,
                                      communcation_dim=16,
                                      use_barys=True,
                                      uv_res=uv_res,
                                      bary_mode='uv',
                                      uv_layers_negative=uv_layers_negative,
                                      uv_layers_positive=uv_layers_positive,
                                      triangle_indices=torch.from_numpy(triangle_indices).long(),
                                      triangle_normals=torch.from_numpy(triangle_normals),
                                      uv_verts=torch.from_numpy(flame_uvs),
                                      uv_vert_inds=torch.from_numpy(flame_uv_vert_inds).long(),
                                      uv_faces=torch.from_numpy(flame_uv_faces).long(),
                                      uv_face_normals=torch.from_numpy(flame_uv_face_normals),
                                      )
        if neutral_only:
            ex_decoder = ZeroDeformation()
        else:
            raise ValueError('Not implemented yet: latent triplane + deformations')
    else:
        raise ValueError(f'Unknown model type {model_type}')

    return id_decoder, ex_decoder

def set_up_codes(cfg,
                 model_type,
                 id_decoder,
                 ex_decoder,
                 train_dataset,
                 val_dataset,
                 \
                 neutral_only : bool = False,
                 include_app : bool = False,
                 variational : bool = False,
                 codebook_numbers_train = None,
                 codebook_numbers_val = None,
                 **kwargs,
                 ):
    # Initializing latent codes.
    modalities = ['geo']
    types = ['vector']
    if model_type == 'nphm':
        n_channels = [id_decoder.lat_dim_glob_geo + id_decoder.lat_dim_loc_geo * (id_decoder.n_anchors + 1)] # [id_decoder.lat_dim]
    elif model_type == 'global':
        n_channels = [cfg['decoder']['id']['lat_dim']]
    elif model_type == 'grid':
        n_channels = [id_decoder.lat_dim * id_decoder.resolution ** 3]
    elif model_type == 'triplane':
        n_channels = [id_decoder.lat_dim * id_decoder.resolution ** 2 * 3]
    elif model_type == 'eg3d':
        n_channels = [id_decoder.lat_dim_app]  # TODO currently generating geo and app together
    elif model_type == 'latent-mesh':
        assert 1 == 2
        if False:
            n_channels = [id_decoder.lat_dim * id_decoder.n_points]
        else:
            n_channels = [id_decoder.lat_dim * uv_res ** 2 * (1 + uv_layers_positive + uv_layers_negative)]

    n_train_geo = len(train_dataset.subjects) if train_dataset is not None else 381
    n_val_geo = len(val_dataset.subjects) if val_dataset is not None else 10
    if codebook_numbers_train is not None:
        n_train_geo = codebook_numbers_train['geo']
    n_latents_train = [n_train_geo]#269] #237]
    n_latents_val = [n_val_geo] #2

    if not neutral_only:
        modalities.append('exp')
        types.append('vector')
        n_channels.append(ex_decoder.lat_dim_expr)
        n_train_exp = len(train_dataset) if train_dataset is not None else 7707
        n_val_exp = len(val_dataset) if train_dataset is not None else 224
        if codebook_numbers_train is not None:
            n_train_exp = codebook_numbers_train['exp']
            n_val_exp = codebook_numbers_val['exp']
        n_latents_train.append(n_train_exp) #5646) #4905)
        n_latents_val.append(n_val_exp) #46 )
    if include_app:
        modalities.append('app')
        types.append('vector')
        if model_type == 'nphm':
            n_channels.append(id_decoder.lat_dim_glob_app + id_decoder.lat_dim_loc_app * (id_decoder.n_anchors + 1))
        elif model_type == 'grid':
            n_channels.append(id_decoder.lat_dim_app * id_decoder.resolution ** 3)
        elif model_type == 'triplane':
            n_channels.append(id_decoder.lat_dim_app * id_decoder.resolution ** 2 * 3)
        elif model_type == 'eg3d':
            n_channels.append(id_decoder.lat_dim_app)
        elif model_type == 'global':
            n_channels.append(id_decoder.lat_dim_app)
        elif model_type == 'latent-mesh':
            if False:
                n_channels.append(id_decoder.lat_dim_app * id_decoder.n_points)
            else:
                n_channels.append(
                    id_decoder.lat_dim_app * uv_res ** 2 * (1 + uv_layers_positive + uv_layers_negative))

        n_train_app = n_train_geo
        n_val_app = n_val_geo
        if codebook_numbers_train is not None:
            n_train_app = codebook_numbers_train['app']
            n_val_app = codebook_numbers_val['app']
        n_latents_train.append(n_train_app) #269) #237 )
        n_latents_val.append(n_val_app) #2)
        if train_dataset is not None and train_dataset.MIRROR:
            n_latents_train = [n*2 for n in n_latents_train]
            n_latents_val = [n*2 for n in n_latents_val]

    latent_codes = LatentCodes(n_latents=n_latents_train,
                                    n_channels=n_channels,  # CFG['decoder']['ex']['lat_dim_ex']],
                                    modalities=modalities,
                                    types=types,
                                    init_zeros=True,
                               variational=variational,
                                    )
    latent_codes_val = LatentCodes(n_latents=n_latents_val,
                                        n_channels=n_channels,  # CFG['decoder']['ex']['lat_dim_ex']],
                                        modalities=modalities,
                                        types=types,
                                        init_zeros=True,
                                   variational=variational,
                                        )

    return latent_codes, latent_codes_val


def set_up_all(cfg,
               args,
               skip_dataset : bool = False,
               rank=None,
               codebook_numbers_train = None,
               codebook_numbers_val = None,
               ):
        # Load stuff required for NPHM.
        #if args.model_type == 'nphm':
        lm_inds = np.load(env_paths.ANCHOR_INDICES_PATH.format(cfg['decoder'].get('decoder_nloc', 65)))
        anchors_path = env_paths.ANCHOR_MEAN_PATH.format(cfg['decoder'].get('decoder_nloc', 65))

        anchors = torch.from_numpy(np.load(anchors_path)).float().unsqueeze(0).unsqueeze(0)
        #else:
        #    lm_inds = None
        #    anchors = None

        if not skip_dataset:
            train_dataset, val_dataset = set_up_datasets(cfg,
                                                         lm_inds=lm_inds,
                                                         **args
                                                         )
        else:
            train_dataset = None
            val_dataset = None

        id_decoder, ex_decoder = set_up_networks(cfg,
                                                 anchors=anchors,
                                                 rank=rank,
                                                 **args
                                                 )

        latent_codes, latent_codes_val = set_up_codes(cfg,
                                                      args.model_type,
                                                      id_decoder,
                                                      ex_decoder,
                                                      train_dataset,
                                                      val_dataset,
                                                      include_app=args.include_app,
                                                      neutral_only=args.neutral_only,
                                                      codebook_numbers_train=codebook_numbers_train,
                                                      codebook_numbers_val=codebook_numbers_val,
                                                      )


        if not skip_dataset:
            print(f'Train Dataset has {len(train_dataset.subjects)} Subjects and {len(train_dataset.subject_IDs)} Expressions')
            print(f'Val Dataset has {len(val_dataset.subjects)} Subjects and {len(val_dataset.subject_IDs)} Expressions')

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'id_decoder': id_decoder,
            'ex_decoder': ex_decoder,
            'latent_codes': latent_codes,
            'latent_codes_val': latent_codes_val,
        }






