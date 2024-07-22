import numpy as np
import pyvista as pv
import trimesh
import cv2
import os
import json
import open3d as o3d
from dphm_tum.utils.io import export_pointcloud_o3d

import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap("jet")

def read_intrinsics_kinect_from_json(path_to_intrinsics_json, im_size=None, center_crop_fix_intrinsics=False, crop_details=None):
    # Kinect recording dumps the intrinsicscalibration to a json
    with open(path_to_intrinsics_json, 'r') as fp:
        print("Loading intrinsics from JSON")
        calibration = json.load(fp)['color']
        # Achtung! these correspond to the original image size it was captured with. cx,cy are in pixels.
        fx = calibration['fx'] 
        fy = calibration['fy'] 
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

def unproject_2d_lms_to_3d(lms_2d, depth_gt, K_inv):
    depth_lms = depth_gt[lms_2d[:, 1].astype(int), lms_2d[:, 0].astype(int)]
    pixels_hom = np.ones([lms_2d.shape[0], 3])
    pixels_hom[:, :2] = lms_2d[:, :2]
    pixels_hom = np.reshape(pixels_hom, [-1, 3])
    lms3D =  pixels_hom.dot( K_inv.T)
    lms_depth = np.reshape(depth_lms, -1)
    print('depth_lms', depth_lms.min(), depth_lms.max())
    lms3D *= lms_depth[:, np.newaxis]
    return lms3D

def unproject_depth_to_points(depth_gt, K_inv):
    xx, yy = np.meshgrid(np.arange(depth_gt.shape[1]), np.arange(depth_gt.shape[0]))
    pixels = np.stack([xx, yy], axis=-1)

    pixels_hom = np.ones([pixels.shape[0], pixels.shape[1], 3])
    pixels_hom[:, :, :2] = pixels
    pixels_hom = np.reshape(pixels_hom, [-1, 3])
    points3D =  pixels_hom.dot( K_inv.T)
    depth = np.reshape(depth_gt, -1)
    
    points3D *= depth[:, np.newaxis]
    return points3D

def calculate_normals(points3D, depth):
    points3D = points3D.reshape(depth.shape[0], depth.shape[1], 3)
    
    # calculate normals
    normals3D = np.zeros_like(points3D, dtype='float32')
    center = points3D[1:-1, 1:-1, :]
    left   = points3D[1:-1, 0:-2, :]
    right  = points3D[1:-1, 2:, :]
    up     = points3D[2:,   1:-1, :]
    down   = points3D[0:-2, 1:-1, :]
    diffX  = right - left
    diffY  = up - down
    Z = np.cross(diffX, diffY) 
    Z_norm = Z / (np.linalg.norm(Z, axis=-1, keepdims=True) )  #+1e-8
    
    du = 0.5 * (depth[1:-1, 2:] - depth[1:-1, 0:-2]) # right -left
    dv = 0.5 * (depth[2:, 1:-1] - depth[0:-2, 1:-1]) # up - down
    
    normals3D[:] = np.nan  # that will set the border to np.nan
    normals3D_center = normals3D[1:-1, 1:-1, :]
    valid = (np.isfinite(center).all(axis=-1)) & (np.isfinite(Z_norm).all(axis=-1)) & (np.isfinite(du)) & (np.isfinite(dv)) & (np.abs(du)<=0.1/2) & (np.abs(dv)<0.1/2)

    normals3D_center[valid, :] = Z_norm[valid, :]
    normals3D[1:-1, 1:-1, :]   = normals3D_center
 
    points3D  = points3D.reshape(-1, 3)
    normals3D = normals3D.reshape(-1, 3)
    return points3D, normals3D, valid

def get_valid_depth_normal_rgb(points3D, normals3D, depth_gt, mask_backgroud=False, near=0.1, far=0.6):
    depth = np.reshape(depth_gt, -1)
    if mask_backgroud:
        valid_region = (np.isfinite(points3D).all(axis=-1)) & (np.isfinite(normals3D).all(axis=-1)) & (depth<far) & (depth>near)  #& (points3D[:, 2] < far) & (points3D[:, 2] > near)
    else:
        valid_region = (np.isfinite(points3D).all(axis=-1)) & (np.isfinite(normals3D).all(axis=-1))
    invalid_region = ~valid_region

    depth[invalid_region] = far
    depth_inv = 1.0 / (depth)
    depth = (depth_inv - depth_inv.min()) / (depth_inv.max() - depth_inv.min())
    normals3D = -normals3D 
    normal = (-normals3D + 1.0)/2.0 * 255
    
    depth_array  =  ( cmap( depth.reshape(depth_gt.shape[0], depth_gt.shape[1]) )[:, :, :3] * 255 ).astype(np.uint8)
    normal_array = normal.reshape(depth_gt.shape[0], depth_gt.shape[1], 3).astype(np.uint8)
    depth_array, normal_array = cv2.cvtColor(depth_array, cv2.COLOR_RGB2BGR), cv2.cvtColor(normal_array, cv2.COLOR_RGB2BGR)
    return depth_array, normal_array, valid_region

def unproject_depth_to_scan_normal_and_lms3d(data_dir = "/cluster/balrog/jtang/Head_tracking/NPHM/dataset/kinect",
                                            subjects = ['student2'], my_record = False, 
                                            color_foldername="color", depth_foldername="depth",
                                            pc_crop_foldername="points",
                                            nms_crop_foldername="normals",
                                            lms_pip_crop_foldername="lms_3d_pip", 
                                            lms_mp_crop_foldername="lms_3d_mp", 
                                            depth_normal_foldername = "depth_normals_bilateral",
                                            height=1080, width = 1920, down_ratio=1,
                                            mask_face_contour = False,
                                            mask_eye_mouth = False, 
                                            near=0.1, far=0.6, 
                                            out_dir = "/cluster/balrog/jtang/Head_tracking/NPHM/dataset/kinect"):
    print('############ Starting Unprojection ############')
    
    for subj in subjects:
        print('Process subject {}'.format(subj))
        data_subj_dir = '{}/{}/'.format(data_dir, subj)
        if my_record:
            intrinsics = np.loadtxt(os.path.join(data_subj_dir, 'camera', 'c00_color_intrinsic.txt')) # changed !!!
            intrinsics_dict = {'color': {
                "cx": intrinsics[0],
                "cy": intrinsics[1],
                "fx": intrinsics[2],
                "fy": intrinsics[3],
                "k1": intrinsics[4],
                "k2": intrinsics[5],
                "k3": intrinsics[6],
                "k4": intrinsics[7],
                "k5": intrinsics[8],
                "k6": intrinsics[9],
                "metric_radius": intrinsics[10],
                "p1": intrinsics[11],
                "p2": intrinsics[12],
                "codx": intrinsics[13], # not sure ???
                "cody": intrinsics[13], # not sure ???
                }
            }
            json_object = json.dumps(intrinsics_dict, indent=4)
    
            # Writing to sample.json
            with open(os.path.join(data_subj_dir, "calibration.json"), "w") as outfile:
                outfile.write(json_object)
        intrinsics = read_intrinsics_kinect_from_json(os.path.join(data_subj_dir, "calibration.json"), (height, width), crop_details=None)

        K = np.eye(3)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]

        K_inv = np.linalg.inv(K)
        print('camera instrinsic: ', K)

        filenames = os.listdir(os.path.join(data_subj_dir, depth_foldername))
        filenames.sort(key=lambda x: int(x.split('.')[0]))
        if down_ratio >1:
            filenames = [f for f in filenames if int(f.split('.')[0]) % down_ratio==0]
        frames = [int(f.split('.')[0]) for f in filenames]
            

        for i, filename in enumerate(filenames):
            depth_gt = cv2.imread(os.path.join(data_subj_dir , depth_foldername , filename), cv2.IMREAD_UNCHANGED)
            depth_gt = depth_gt / 1000 # to metres
            print('depth :', depth_gt.min(), depth_gt.max(), depth_gt.shape)
            color_gt = cv2.imread(os.path.join(data_subj_dir, color_foldername, filename))
            print('color :', color_gt.shape)

            lms_2d_pip = np.load('{}/{}/PIPnet_landmarks/{:05d}.npy'.format(data_subj_dir, color_foldername, frames[i]))
            print('lms_2d_pip:', lms_2d_pip[:, 0].min(), lms_2d_pip[:, 0].max(), lms_2d_pip[:, 1].min(), lms_2d_pip[:, 1].max())

            lms_2d = np.load('{}/{}/Mediapipe_landmarks/{:05d}.npy'.format(data_subj_dir, color_foldername,  frames[i]))[:, :2]
            print('lms_2d mp:', lms_2d[:, 0].min(), lms_2d[:, 0].max(), lms_2d[:, 1].min(), lms_2d[:, 1].max())
            lms_2d[:, 0] *= width
            lms_2d[:, 1] *= height
            
            gray = cv2.cvtColor(color_gt, cv2.COLOR_BGR2GRAY)
            if mask_face_contour:
                mask = np.zeros_like(gray)
                # jawline 
                cv2.fillPoly(mask, [lms_2d[(21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 
                                            152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162), :].astype(int)], (255))
                #mouth
                mask = np.reshape(mask, (-1))
                masked_contour = (mask == 255)
            else:
                mask = np.zeros_like(gray)
                mask = np.reshape(mask, (-1))
                masked_contour = (mask == 0)

            if mask_eye_mouth:
                gray = cv2.cvtColor(color_gt, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(gray)

                #left eye
                cv2.fillPoly(mask, [lms_2d[(33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7), :].astype(int)], (255))
                #right eye
                cv2.fillPoly(mask, [lms_2d[(263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249), :].astype(int)], (255))
                #mouth
                cv2.fillPoly(mask, [lms_2d[(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95), :].astype(int)], (255))
        
                mask = np.reshape(mask, (-1))
                masked_eye_mouth = (mask != 255)
            else:
                gray = cv2.cvtColor(color_gt, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(gray)
                mask = np.reshape(mask, (-1))
                masked_eye_mouth = (mask != 255)

            # pipnet landmark unprojection
            lms3D_pip = unproject_2d_lms_to_3d(lms_2d_pip, depth_gt, K_inv)
            # media pip landmark unprojection
            lms3D = unproject_2d_lms_to_3d(lms_2d, depth_gt, K_inv)
            
            # back project depth map to points
            print('depth', depth_gt.shape)
            points3D = unproject_depth_to_points(depth_gt, K_inv)
            points3D, normals3D, _ = calculate_normals(points3D, depth_gt)
        
            depth_bilateral = cv2.bilateralFilter(depth_gt.astype(np.float32), 9, 200, 200) 
            print('depth bilateral', depth_bilateral.shape)
            points3D_bilateral = unproject_depth_to_points(depth_bilateral, K_inv)
            
            points3D_bilateral, normals3D_bilateral, _ = calculate_normals(points3D_bilateral, depth_bilateral)
            depth_array, normal_array, valid_region = get_valid_depth_normal_rgb(points3D, normals3D, depth_gt, mask_backgroud=True, near=near, far=far)
            depth_array_bilateral, normal_array_bilateral, valid_region_bilateral = get_valid_depth_normal_rgb(points3D_bilateral, normals3D_bilateral, depth_bilateral, mask_backgroud=True, near=near, far=far)
            
            out_dir = '{}/{}/'.format(data_dir, subj)
            os.makedirs(out_dir + depth_normal_foldername, exist_ok=True)
            mask = (valid_region  & (points3D[:, 2]>near) & (points3D[:, 2]<far)).reshape(color_gt.shape[0], color_gt.shape[1]) 
            depth_array[~mask] = 255
            normal_array[~mask] = 255
            color_masked = np.ones((color_gt.shape[0], color_gt.shape[1], 3), dtype='uint8') * 255
            color_masked[mask, :]  = color_gt[mask, :]
            cv2.imwrite(out_dir + '{}/{:05d}_depth.jpg'.format(depth_normal_foldername, i), depth_array)
            cv2.imwrite(out_dir + '{}/{:05d}_normal.jpg'.format(depth_normal_foldername, i), normal_array)
            cv2.imwrite(out_dir + '{}/{:05d}_colormask.jpg'.format(depth_normal_foldername, i), color_masked )
            

            mask_filter = (valid_region_bilateral & (points3D_bilateral[:, 2]>near) & (points3D_bilateral[:, 2]<far)).reshape(color_gt.shape[0], color_gt.shape[1]) 
            depth_array_bilateral[~mask_filter] = 255
            normal_array_bilateral[~mask_filter] = 255
            color_masked_filter = np.ones((color_gt.shape[0], color_gt.shape[1], 3), dtype='uint8') * 255
            color_masked_filter[mask_filter, :]  =  color_gt[mask_filter, :]
            cv2.imwrite(out_dir + '{}/{:05d}_depth_filter.jpg'.format(depth_normal_foldername, i), depth_array_bilateral)
            cv2.imwrite(out_dir + '{}/{:05d}_normal_filter.jpg'.format(depth_normal_foldername, i), normal_array_bilateral)
            cv2.imwrite(out_dir + '{}/{:05d}_colormask_filter.jpg'.format(depth_normal_foldername, i), color_masked_filter)
        

            valid_union = valid_region & valid_region_bilateral &  masked_contour & masked_eye_mouth & (points3D[:, 2]>near) & (points3D[:, 2]<far)
            #!!!!! points3D_final = points3D_bilateral[valid_union] 
            points3D_final = points3D[valid_union] 
            #!!!!!
            normals3D_final = -normals3D_bilateral[valid_union]
            print(i, points3D_final.shape, normals3D_final.shape)
            assert normals3D_final.shape == points3D_final.shape

            out_dir = '{}/{}/'.format(data_dir, subj)
            os.makedirs(out_dir + pc_crop_foldername, exist_ok=True)
            os.makedirs(out_dir + nms_crop_foldername, exist_ok=True)
            os.makedirs(out_dir + lms_pip_crop_foldername, exist_ok=True)
            os.makedirs(out_dir + lms_mp_crop_foldername, exist_ok=True)
            
            index = i # frames[i]
            np.save(out_dir + '{}/{:05d}.npy'.format(pc_crop_foldername, index), points3D_final)
            np.save(out_dir + '{}/{:05d}.npy'.format(nms_crop_foldername, index), normals3D_final)
            np.save(out_dir + '{}/{:05d}.npy'.format(lms_pip_crop_foldername, index), lms3D_pip)
            np.save(out_dir + '{}/{:05d}.npy'.format(lms_mp_crop_foldername, index), lms3D)
            
            rand_idx = np.random.randint(0, points3D_final.shape[0], size=10000)
            export_pointcloud_o3d(out_dir + '{}/{:05d}.ply'.format(pc_crop_foldername, index), points3D_final[rand_idx], normals=normals3D_final[rand_idx])
            export_pointcloud_o3d(out_dir + '{}/{:05d}.ply'.format(lms_pip_crop_foldername, index), lms3D_pip)
            export_pointcloud_o3d(out_dir + '{}/{:05d}.ply'.format(lms_mp_crop_foldername, index), lms3D)
            
if __name__ == '__main__':      
    subjects =[ 
        "aria_talebizadeh_eyeblink", "aria_talebizadeh_smile", "aria_talebizadeh_fastalk", "aria_talebizadeh_mouthmove", "aria_talebizadeh_rotatemouth", 
        "arnefucks_eyeblink",   "arnefucks_smile",  "arnefucks_fastalk",  
        "arnefucks_mouthmove",  "arnefucks_rotatemouth", 
        "elias_wohlgemuth_eyeblink",  "elias_wohlgemuth_smile",  "elias_wohlgemuth_fastalk",  "elias_wohlgemuth_mouthmove",  "elias_wohlgemuth_rotatemouth", 
        "innocenzo_fulgintl_eyeblink",   "innocenzo_fulgintl_smile",  "innocenzo_fulgintl_fastalk",  "innocenzo_fulgintl_mouthmove",  "innocenzo_fulgintl_rotatemouth", 
        "mahabmarhai_eyeblink", "mahabmarhai_smile",  "mahabmarhai_fastalk",  "mahabmarhai_mouthmove",  "mahabmarhai_rotatemouth", 
        "manuel_eyeblink",  "manuel_smile",  "manuel_fastalk",  "manuel_mouthmove",  "manuel_rotatemouth",
        "michaeldyer_eyeblink2",  "michaeldyer_smile2",  "michaeldyer_fastalk2",  "michaeldyer_mouthmove2",  "michaeldyer_rotatemouth2", 
        "seddik_houimli_eyeblink",  "seddik_houimli_smile",  "seddik_houimli_fastalk",  "seddik_houimli_mouthmove",  "seddik_houimli_rotatemouth", 
        "weicao_random2", "weicao_angry",  "weicao_mouthmove", "weicao_mouthmovelarge", "weicao_fastalk",  "weicao_rotatemouth", "weicao_talk",  "weicao_smile2",  "weicao_smile", 
        "changluo_random2", "changluo_fastalk", "changluo_mouthmovelarge", "changluo_rotatemouth", "changluo_angry", "changluo_talk", "changluo_smile",
        "haoxuan_eyeblink", "haoxuan_smile", "haoxuan_fastalk", "haoxuan_mouthmove", "haoxuan_rotatemouth", 
        "siyunliang_eyeblink", "siyunliang_smile", "siyunliang_fastalk", "siyunliang_mouthmove", "siyunliang_rotatemouth",
        "ali_kocal_mouthmove", "ali_kocal_rotatemouth", "ali_kocal_eyeblink",  "ali_kocal_smile", "ali_kocal_fastalk", 
        "christoph_mouthmove", "christoph_rotatemouth", "christoph_eyeblink",  
        "christoph_smile", "christoph_fastalk",  
        "felix_mouthmove", "felix_rotatemouth", "felix_eyeblink",  "felix_smile", "felix_fastalk",  
        "honglixu_mouthmove", "honglixu_rotatemouth", "honglixu_eyeblink",  "honglixu_smile", "honglixu_fastalk",  
        "madhav_agarwal_mouthmove", "madhav_agarwal_rotatemouth", "madhav_agarwal_eyeblink",  "madhav_agarwal_smile", "madhav_agarwal_fastalk",  
        "medhansh_mouthmove", "medhansh_eyeblink",  "medhansh_smile", "medhansh_fastalk",   "medhansh_rotatemouth", 
        "mohak_mouthmove",  "mohak_rotatemouth", "mohak_eyeblink",  "mohak_smile",  "mohak_fastalk",  
        "mykola_mouthmove", "mykola_rotatemouth", "mykola_eyeblink",  "mykola_smile", "mykola_fastalk", 
        "umur_gogebakan_mouthmove",  "umur_gogebakan_rotatemouth",  "umur_gogebakan_fastalk", "umur_gogebakan_eyeblink",   "umur_gogebakan_smile", 
        "nikolas_mouthmove",  "nikolas_rotatemouth",  "nikolas_fastalk",  "nikolas_eyeblink",   "nikolas_smile", 
        "viet_mouthmove",   "viet_rotatemouth",  "viet_fastalk",  "viet_eyeblink",   "viet_smile", 
    ]
    
    subjects =[ 
        #
        "leni_rohe_eyeblink",
        #"leni_rohe_mouthmove", "leni_rohe_rotatemouth", "leni_rohe_smile", 
        #"neha_rao_eyeblink", "neha_rao_mouthmove",  "neha_rao_rotatemouth", "neha_rao_smile", 
    ]
    
    parser = argparse.ArgumentParser(
        description='Run generation'
    )
    parser.add_argument('-data_dir', default='/cluster/balrog/jtang/Head_tracking/NPHM/dataset/DPHM-Kinect/', type=str)
    parser.add_argument('-near', default=0.1, type=float)
    parser.add_argument('-far',  default=0.6, type=float)
    args = parser.parse_args()
    
    unproject_depth_to_scan_normal_and_lms3d(data_dir=args.data_dir, subjects = subjects, my_record = True,  near=args.near, far=args.far)
    
