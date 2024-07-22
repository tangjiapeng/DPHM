import numpy as np
import torch

# convert the scan 
def delete_nan_and_inside_kinect(point_cloud_np, normals_np,  R, t, c, flame2scan=False, raw_scan=False, raw_scan2=False):
    valid = np.logical_not(np.any(np.isnan(point_cloud_np), axis=-1))
    valid_n = np.logical_not(np.any(np.isnan(normals_np), axis=-1))
    valid = np.logical_and(valid, valid_n)
    point_cloud_np = point_cloud_np[valid, :]
    normals_np = normals_np[valid, :]

    if flame2scan:
        point_cloud_np_nphm = 4 * (point_cloud_np - t) @ R / c
        normals_np_nphm = normals_np @ R
    else:
        point_cloud_np_nphm = 4 * (c * point_cloud_np @ R.T + t)
        normals_np_nphm = normals_np @ R.T 
        
    if raw_scan:
        inside = np.logical_and(np.logical_and(point_cloud_np_nphm[:, 1] > -0.3, point_cloud_np_nphm[:, 2] > -0.1), point_cloud_np_nphm[:, 2] < 0.5) 
    elif raw_scan2:
        inside = np.logical_and(np.logical_and(point_cloud_np_nphm[:, 1] > -0.45, point_cloud_np_nphm[:, 2] > -0.1), point_cloud_np_nphm[:, 2] < 0.5) 
    else:
        inside = np.logical_and(np.logical_and(point_cloud_np_nphm[:, 1] > -0.4, point_cloud_np_nphm[:, 2] > 0.0), point_cloud_np_nphm[:, 2] < 0.5)
    point_cloud_np = point_cloud_np[inside]
    normals_np = normals_np[inside]
    point_cloud_np_nphm = point_cloud_np_nphm[inside]
    normals_np_nphm = normals_np_nphm[inside]
    return point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm


def random_sample(point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm, num_points=100000):
    rnd_ind = torch.randint(0, point_cloud_np.shape[0], size=[num_points])
    point_cloud_np = point_cloud_np[rnd_ind, :]
    normals_np = normals_np[rnd_ind, :]
    point_cloud_np_nphm = point_cloud_np_nphm[rnd_ind, :]
    normals_np_nphm = normals_np_nphm[rnd_ind, :]
    return point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm


def add_gaussian_noise(point_cloud_np, normals_np, noise_level=0.0):
    point_cloud_np_noise = point_cloud_np + np.random.normal(loc=0.0, scale=noise_level, size=(point_cloud_np.shape[0], point_cloud_np.shape[1]))
    normals_np_noise = normals_np + np.random.normal(loc=0.0, scale=noise_level, size=(normals_np.shape[0], normals_np.shape[1]))
    normals_np_noise = normals_np_noise / np.linalg.norm(normals_np_noise, axis=1, keepdims=True)
    return point_cloud_np_noise, normals_np_noise

