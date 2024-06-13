import os
import os.path
from dreifus.matrix import Pose
import numpy as np
from typing import Dict
from famudy.data import FamudySequenceDataManager
from famudy.constants import SERIALS
from nphm_tum.utils.tranformations import invert_similarity_transformation, apply_transform
from nphm_tum import env_paths

def get_transform_mvs2flame2nphm(p_id, seq_name,
                             flame_params_path : str,
                             corrective_transform_path : str,
                             frame : int,
                             inverse : bool = False,
                             nphm_tracking_dir = None,
                             return_rotation_only : bool = False,
                             not_in_nphm_dataset : bool = False,
                             skip_corrective : bool = False,
                             ignore_scale : bool = False,
                             ):

    tracked_flame_params = np.load(flame_params_path)
    frame_idx = np.where(tracked_flame_params['frames'] == frame)[0].item()

    has_corrective = False
    if os.path.exists(corrective_transform_path) and not skip_corrective:
        if not not_in_nphm_dataset:
            tracked_corrective_transform = np.load(corrective_transform_path)
            has_corrective = True

    #if not skip_corrective:
    #    assert has_corrective

    s = tracked_flame_params['scale']
    if ignore_scale:
        s = 1
    R = tracked_flame_params['rotation_matrices'][frame_idx, :, :]
    t = tracked_flame_params['translation'][frame_idx, :]

    T_flame2mvs = np.eye(4)
    if return_rotation_only:
        T_flame2mvs[:3, :3] = R
    else:
        T_flame2mvs[:3, :3] = s*R
        T_flame2mvs[:3, 3] = t
    T_mvs2flame = invert_similarity_transformation(T_flame2mvs)


    T_flame2nphm = np.eye(4)
    if not not_in_nphm_dataset and has_corrective and not skip_corrective:
        frame_idx = 0
        s = tracked_corrective_transform['scale'][frame_idx]
        if ignore_scale:
            s = 1
        R = tracked_corrective_transform['rotation'][frame_idx, :, :]
        t = tracked_corrective_transform['translation'][frame_idx, :]

        if return_rotation_only:
            T_flame2nphm[:3, :3] = R
        else:
            T_flame2nphm[:3, :3] = s*R
            T_flame2nphm[:3, 3] = t

    combined_transform =  T_flame2nphm @ T_mvs2flame
    if not return_rotation_only:
        if not ignore_scale:
            combined_transform = 4*np.eye(4) @ combined_transform
    if inverse:
        combined_transform = invert_similarity_transformation(combined_transform)

    if nphm_tracking_dir is not None:
        for nphm_tracking_folder in env_paths.nphm_tracking_name_priorities:
            src_dir = env_paths.NERSEMBLE_DATASET_PATH + f'/{p_id:03d}/sequences/{seq_name}/annotations/tracking/{nphm_tracking_folder}/'
            if os.path.exists(src_dir):
                break
        if not os.path.exists(src_dir):
            raise ValueError(f'Could not find NPHM tracking for participant {p_id} and sequence {seq_name}')
        sim_mvs2nphm = np.load(src_dir + '/{:05d}_corrective_mvs2nphm_fine.npy'.format(frame))
        sim_nphm2mvs = invert_similarity_transformation(sim_mvs2nphm)
        combined_transform = combined_transform @ sim_nphm2mvs
    return combined_transform


def load_camera_params(mv_manager:
                       FamudySequenceDataManager,
                       is_nag : bool = False,
                       downscale_factor : float = None,
                       ):
    calibration_result = mv_manager.load_calibration_result()
    # get cam2world poses in OpenCV convention
    cam_to_world_poses = [Pose(c).invert() for c in calibration_result.params_result.get_poses()]

    intrinsics = mv_manager.load_calibration_result().params_result.get_intrinsics()

    if downscale_factor is None:

        intrinsics.rescale(0.5)
    elif downscale_factor is not None:

        intrinsics.rescale(downscale_factor)
    c2w_poses = {}
    Ks = {}
    for cam_id in range(16):
        c2w_poses[SERIALS[cam_id]] = cam_to_world_poses[cam_id]
        Ks[SERIALS[cam_id]] = intrinsics
    return Ks, c2w_poses