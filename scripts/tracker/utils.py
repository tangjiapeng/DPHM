import os
import numpy as np
import torch

def load_tracking_parameters(file_path, device):
    lat_dict = np.load(file_path)
    lat_rep_shape = torch.from_numpy(lat_dict["lat_rep_shape"]).float().to(device)
    lat_reps_expr = torch.from_numpy(lat_dict["lat_reps_expr"]).float().to(device)
    trans = torch.from_numpy(lat_dict["trans"]).float().to(device)
    rot = torch.from_numpy(lat_dict["rot"]).float().to(device)
    rot_euler = torch.from_numpy(lat_dict["rot_euler"]).float().to(device)
    scale = torch.from_numpy(lat_dict["scale"]).float().to(device)
    return lat_reps_expr, lat_rep_shape, trans, rot, rot_euler, scale

def save_tracking_parameters(filepath, lat_reps_expr, lat_rep_shape, rigid_transform_dict):
    np.savez(filepath, lat_reps_expr=lat_reps_expr.detach().cpu().numpy(),
                                            lat_rep_shape=lat_rep_shape.detach().cpu().numpy(),
                                            rot=rigid_transform_dict["rot"].detach().cpu().numpy(), 
                                            trans=rigid_transform_dict["trans"].detach().cpu().numpy(),
                                            scale=rigid_transform_dict["scale"].detach().cpu().numpy(),
                                            rot_euler=rigid_transform_dict["rot_euler"].detach().cpu().numpy(), 
    )
    
    
def transform_points_from_nphm_to_scan_space(points_nphm, c, rot, t, flame2scan=False):
    if flame2scan:
        points_scan = c * (points_nphm/4.0) @ (rot.T)  + t
    else:
        points_scan = ((points_nphm/4.0 - t)/c ).dot(rot)
    return points_scan


def transform_points_from_scan_to_nphm_space(points_scan, c, rot, t, flame2scan=False):
    if flame2scan:
        points_nphm = 4 * (points_scan - t) @ rot / c
    else:
        points_nphm = (c * points_scan @rot.T + t) * 4.0 
    return points_nphm


    