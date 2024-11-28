import sys
sys.path.append("/rhome/jtang/3d_recon/PVD")
from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
cham3D = chamfer_3DDist()

import numpy as np
import os
import os.path as osp
import pyvista as pv
import trimesh
import torch
from glob import glob
import cv2 as cv
from PIL import Image

def sample_points(mesh, npoints=10000):
    # mainly for flame
    _, face_idx = mesh.sample(npoints, return_index=True)
    alpha = np.random.dirichlet((1,)*3, npoints)
    
    vertices = mesh.vertices
    faces = mesh.faces
    v = vertices[faces[face_idx]]
    normals = mesh.face_normals[face_idx]
    points = (alpha[:, :, None] * v).sum(axis=1)
    return points, normals

def calcaute_nearsest_distance(recon_points, observed_points, delta_dist=0.02, return_idx=False):
    device = torch.device("cuda")
    if True:
        recon_points_tensor = torch.from_numpy(recon_points).float().to(device)[None, :]
        observed_points_tensor = torch.from_numpy(observed_points).float().to(device)[None, :]
        accuracy, completeness, idx1, idx2 =  cham3D(recon_points_tensor, observed_points_tensor)
        dists = np.sqrt((accuracy).cpu().numpy()).reshape(-1)
    else:
        tree = KDTree(observed_points)
        dists, _ = tree.query(recon_points, k=1)
    
    dists_raw = dists.copy()
    
    print('nndists:', dists.min(), dists.max())
    invalid_max = (dists > delta_dist)
    invalid_min = (dists <= 0.0001)
    dists[invalid_max] = delta_dist
    dists[invalid_min] = 0.
    
    if return_idx:
        return dists, dists_raw, idx1.cpu().numpy()
    else:
        return dists

def calculate_normal_cosin(recon_normals, observed_normals, idx):
    recon_normals = recon_normals / np.linalg.norm(recon_normals, axis=1, keepdims=True)
    observed_normals = observed_normals /np.linalg.norm(observed_normals, axis=1, keepdims=True)
    
    observed_normals_matched = observed_normals[idx]
    normal_cosine = (recon_normals * observed_normals_matched).sum(axis=-1)
    normal_cosine = np.clip(normal_cosine, 0, 1)
    normal_error =  1 - normal_cosine
    return normal_cosine, normal_error


def transform_points_world2cam(p_world, n_world, camera_extrinsics):
    p_world_hom = np.hstack([p_world, np.ones((p_world.shape[0], 1))])
    transform_world2cam  = np.linalg.inv(camera_extrinsics)
    p_cam = p_world_hom @ transform_world2cam.T
    
    ns = np.ones_like(p_cam)
    ns[:, :3] = n_world
    n_cam = ns[:, :3] @ transform_world2cam[:3, :3].T

    points3d_cam = p_cam[:, :3]
    normals3d_cam = n_cam[:, :3]
    return points3d_cam, normals3d_cam


def cut_thorat(mesh, z_min=-0.15):
    mesh_vert_mask = mesh.vertices[:, 2] > z_min
    face_mask = mesh_vert_mask[mesh.faces].all(axis=1)
            
    mesh_cropface = mesh.copy()
    mesh_cropface.update_vertices(mesh_vert_mask)
    mesh_cropface.update_faces(face_mask)
    return mesh_cropface
    

def render_snapshot(save_path,  mesh=None, mesh_errors=None, points=None, points_errors=None, normals=None, labels=None, vis_mesh=False, vis_points=False, color='lightblue', nphm_coord=False, opengl_coord=False, black=False):
    if black:
        pv.global_theme.font.color = 'black'
    else:
        pv.global_theme.background = 'white'
    #pv.global_theme.color = 'white'
    #pv.set_plot_theme('document')
    pl = pv.Plotter(off_screen=True)
    if vis_points:
        if points_errors is not None:
            pl.add_points(points, render_points_as_spheres=True, point_size=2.5, scalars=points_errors,  cmap='jet', scalar_bar_args={'title': 'mesh2surf error',  'fmt':'%10.4f',})
            
        elif normals is not None:
            normals_rgb = ( (normals + 1.0)/2.0 *255).astype(int) #np.uint8)
            pl.add_points(points, render_points_as_spheres=True, point_size=2.5, scalars=normals_rgb, rgb=True)
            
        elif labels is not None:
            #[str(i) for i in range(points.shape[0])]
            pl.add_point_labels(points, labels, render_points_as_spheres=True, point_size=2.5, point_color=color) 
            
        else:
            pl.add_points(points, render_points_as_spheres=True, point_size=2.5, color=color) #scalars=points[:, 2])
        
    if vis_mesh:
        if mesh_errors is not None:
            pl.add_mesh(mesh, scalars=mesh_errors, cmap='jet', scalar_bar_args={'title': 'mesh2surf error',  'fmt':'%10.4f',})
                                                                #    'title_font_size':35,
                                                                # 'label_font_size':30, 'outline': True, 'fmt':'%10.2f',})
        else:
            pl.add_mesh(mesh, color=color)
        
    pl.reset_camera()
    if nphm_coord:
        pl.camera.position = (0, 0, 3)
        pl.camera.zoom(1.4)
        pl.set_viewup((0, 1, 0))
        pl.camera.view_plane_normal = (-0, -0, 1)
    elif opengl_coord:
        pl.camera.position = (0, 0, 0)
        pl.camera.zoom(1.4)
        pl.set_viewup((0, 1, 0))
        pl.camera.view_plane_normal = (0, 0, -1)
    else:
        pl.camera.position = (0, 0, 0)
        pl.camera.zoom(1.4)
        pl.set_viewup((0, -1, 0))
        pl.camera.view_plane_normal = (-0, -0, 1)
    pl.show(screenshot=save_path)


def convert_opengl_coord_to_opencv_coord(points):
    points_opencv = np.stack([ points[:, 0], -points[:, 1], -points[:, 2]], axis=1)
    return points_opencv
    
def render_from_camera(save_path, intrinsics=(914.2612915039063, 914.2612915039063, 964.8820190429688, 541.9856567382813),
                       mesh=None, mesh_errors=None, points=None, points_errors=None, anchors=None, normals=None, labels=None, vis_mesh=False, vis_points=False, vis_anchors=False, color='lightblue', down_ratio=1, black=False, transparent=False): 
    color_width, color_height = 1920, 1080
    IMG_W, IMG_H = color_width = int(color_width / down_ratio), int(color_height / down_ratio)
    
    if points is not None or  points_errors is not None or mesh_errors is not None:
        lighting='none'
    else:
        lighting='three lights'
    
    p = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H], lighting=lighting) # default: lighting: 'light kit'
    if transparent or save_path.endswith('.png'):
        pv.rcParams['transparent_background'] = True
    else:
        if black:
            p.background_color = (0, 0, 0, 0) #black 
        else:
            p.background_color = (255, 255, 255, 255) #white
            #p.background_color = (255, 255, 255, 0) # also white  ??
    
    # add objects
    if vis_points:
        if points_errors is not None:
            p.add_points(points, render_points_as_spheres=True, point_size=2.5, scalars=points_errors, cmap='jet',  scalar_bar_args={'title': 'mesh2surf error',  'fmt':'%10.4f',})
            
        elif normals is not None:
            if save_path.endswith('.png'):
                #normals_rgb = ((normals[:, (2, 1, 0,)]  + 1.0)/2.0 *255).astype(int) #np.uint8)
                normals_rgb = ((-normals  + 1.0)/2.0 *255).astype(int) #np.uint8)
            else:
                #normals_rgb = ((normals  + 1.0)/2.0 *255).astype(int) #np.uint8)
                normals_rgb = ((normals[:, (2, 1, 0,)]  + 1.0)/2.0 *255).astype(int) #np.uint8) ##????--> green
                
                normals_rgb = ((-normals  + 1.0)/2.0 *255).astype(int) #??? --> purple
            p.add_points(points, render_points_as_spheres=True, point_size=2.5, scalars=normals_rgb, rgb=True)
        else:
            p.add_points(points, render_points_as_spheres=True, point_size=2.5, color=color)
        
    if vis_mesh:
        if mesh_errors is not None:
            p.add_mesh(mesh, scalars=mesh_errors, cmap='jet', scalar_bar_args={'title': 'mesh2surf error',  'fmt':'%10.4f',})
                                                                #    'title_font_size':35,
                                                                # 'label_font_size':30, 'outline': True, 'fmt':'%10.2f',})
        else:
            p.add_mesh(mesh, color=color)
        
    if vis_anchors:
        if labels is not None:
            #[str(i) for i in range(anchors.shape[0])]
            p.add_point_labels(anchors, labels, render_points_as_spheres=True, point_size=10, point_color='red')
        else:
            p.add_points(anchors, render_points_as_spheres=True, point_size=10, color='red')
            
    # if points_errors is not None or mesh_errors is not None:
    #     pass
    # else:
    #     light_position = np.zeros([1, 4])
    #     light_position[0, 2: ] = 1
    #     # light_position = (camera_extrinsics @ light_position.T).T
    #     light_position = light_position[:, :3]
    #     light = pv.Light(position=(light_position[0, 0], light_position[0, 1], light_position[0, 2]), show_actor=False, positional=True, cone_angle=30, exponent=20, intensity=1.0)
    #     p.add_light(light)
    
    # pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    # pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    # pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)

    # ----------------------------------------------------------
    # Intrinsics
    # ----------------------------------------------------------
    # cx = intrinsics.cx
    # cy = intrinsics.cy
    # fx = intrinsics.fx
    # fy = intrinsics.fy
    
    w = p.window_size[0]
    h = p.window_size[1]
    
    # camera intrinsics of kinect
    cx = intrinsics[2]
    cy = intrinsics[3]
    fx = intrinsics[0]
    fy = intrinsics[1]

    # convert the principal point to window center (normalized coordinate system) and set it
    wcx = -2 * (cx - float(w) / 2) / w
    wcy = 2 * (cy - float(h) / 2) / h
    p.camera.SetWindowCenter(wcx, wcy)

    # convert the focal length to view angle and set it
    view_angle = 180 / np.pi * (2.0 * np.arctan2(h / 2.0, fx)) 
    p.camera.SetViewAngle(view_angle)

    # ----------------------------------------------------------
    # Extrinsics
    # ----------------------------------------------------------

    # apply the transform to scene objects
    # vtk_pose = vtk.vtkMatrix4x4()
    # for i in range(pose.shape[0]):
    #     for j in range(pose.shape[1]):
    #         vtk_pose.SetElement(i, j, pose[i, j])

    # p.camera.SetModelTransformMatrix(vtk_pose)

    # the camera can stay at the origin because we are transforming the scene objects
    p.camera.SetPosition(0, 0, 0)

    # look in the +Z direction of the camera coordinate system
    p.camera.SetFocalPoint(0, 0, 1)

    # the camera Y axis points down
    p.camera.SetViewUp(0, -1, 0)
    
    # ensure the relevant range of depths are rendered
    p.renderer.ResetCameraClippingRange()
    #pl.camera.clipping_range = (0.1, 2)
    
    p.render()  # Important, otherwise view isn't updated when render_from_camera() is called multiple times
    
    img = np.asarray(p.screenshot(transparent_background=True)) 
    print('RENDER SCREENSHOT:', img.shape)
    
    p.close()
    
    if transparent or save_path.endswith('.png'):    
        I = Image.fromarray(img)
    else:
        I = Image.fromarray(img[:, :, :3])
    I.save(save_path)
    return img


def blend_color_and_recon(color_path, recon_path, blend_path, ratio=0.7, black_background=False):
    # read two original images
    color = cv.imread(color_path)
    recon = cv.imread(recon_path)
    # composite/blende two images
    if black_background: 
        mask = (recon == 0).all(axis=-1)[:, :, None]
    else:
        mask = (recon >= 200).all(axis=-1)[:, :, None]
    background = color *  mask
    forground = color * (1 - mask) * (1 - ratio) + recon * (1 - mask) * ratio
    blend_img = background + forground
    fore_img = color * mask 
    back_img = recon * (1 - mask)
    # save composited img
    cv.imwrite(blend_path, blend_img)

import json
def read_intrinsics_kinect_from_json(path_to_intrinsics_json, im_size=None, center_crop_fix_intrinsics=False, crop_details=None):
    # Kinect recording dumps the intrinsicscalibration to a json
    with open(path_to_intrinsics_json, 'r') as fp:
        print("Loading intrinsics from JSON")
        calibration = json.load(fp)['color']
        # these correspond to the original image size it was captured with. cx,cy are in pixels.
        fx = calibration['fx'] * 1
        fy = calibration['fy'] * 1
        cx = calibration['cx']
        cy = calibration['cy']
        print("old cx cy: %d %d" % (cx, cy))

        if crop_details is not None:
            print("cropping images, adapting intrinsics")
            h, w = im_size
            crop_start=crop_details['start']
            crop_size = crop_details['crop_size']

            cx = calibration['cx'] - crop_start[1]
            cy = calibration['cy'] - crop_start[0]
            print("new cx cy: %d %d" % (cx, cy))
    print("Done.")

    return np.array([fx,fy,cx,cy])