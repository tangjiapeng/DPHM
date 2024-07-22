import numpy as np
import trimesh
from typing import Union

def invert_similarity_transformation(T):
    inverse_rotation_scaling = np.linalg.inv(T[:3, :3])
    inverse_translation = inverse_rotation_scaling @ -T[:3, 3]
    inverse_transform = np.eye(4)
    inverse_transform[:3, :3] = inverse_rotation_scaling
    inverse_transform[:3, 3] = inverse_translation
    return inverse_transform


def transform_mvs2flame2nphm(flame_params_path, corrective_transform_path, frame_idx, from_flame_to_mvs=True, inverse=False, return_rotation_only=False):

    tracked_flame_params = np.load(flame_params_path)
    tracked_corrective_transform = np.load(corrective_transform_path)

    s = tracked_flame_params['scale']
    R = tracked_flame_params['rotation_matrices'][frame_idx, :, :]
    t = tracked_flame_params['translation'][frame_idx, :]

    if from_flame_to_mvs:
        # the parameter are used to transform flame mesh to mvs space.  default True used in shivangi's code
        T_flame2mvs = np.eye(4)
        if return_rotation_only:
            T_flame2mvs[:3, :3] = R
        else:
            T_flame2mvs[:3, :3] = s*R
            T_flame2mvs[:3, 3] = t
        T_mvs2flame = invert_similarity_transformation(T_flame2mvs)
    else:
        # the parameter are used to transform mvs coordinate to flame space: used in simon's code
        T_mvs2flame = np.eye(4)
        if return_rotation_only:
            T_mvs2flame[:3, :3] = R
        else:
            T_mvs2flame[:3, :3] = s*R
            T_mvs2flame[:3, 3] = t

    s = tracked_corrective_transform['scale'][frame_idx]
    R = tracked_corrective_transform['rotation'][frame_idx, :, :]
    t = tracked_corrective_transform['translation'][frame_idx, :]

    T_flame2nphm = np.eye(4)
    if return_rotation_only:
        T_flame2nphm[:3, :3] = R
    else:
        T_flame2nphm[:3, :3] = s*R
        T_flame2nphm[:3, 3] = t

    combined_transform =  T_flame2nphm @ T_mvs2flame
    if not return_rotation_only:
        combined_transform = 4*np.eye(4) @ combined_transform
    if inverse:
        combined_transform = invert_similarity_transformation(combined_transform)

    return combined_transform


def apply_transform(object3d : Union[trimesh.Trimesh, np.ndarray], transform: np.ndarray):

    if isinstance(object3d, trimesh.Trimesh):
        points3d = object3d.vertices
    else:
        points3d = object3d

    points3d_hom = np.concatenate([points3d, np.ones_like(points3d[:, :1])], axis=1)

    points3d_prime = (transform @ points3d_hom.T).T
    points3d_prime = points3d_prime[:, :3] # remove homogeneous coordinates
    if isinstance(object3d, trimesh.Trimesh):
        object3d.vertices = points3d_prime
    else:
        object3d = points3d_prime

    return object3d


def rigid_transform(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return 1, R, np.squeeze(t)


def similarity_transform(from_points, to_points):
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)

    return c, R, t

