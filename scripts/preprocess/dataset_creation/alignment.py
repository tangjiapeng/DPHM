import numpy as np
import pyvista as pv
from .coordinate_transform import rigid_transform, similarity_transform

FLAME_LANDMARK_INDICES = np.array([2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637, 3587, 3582, 3580, 3756, 2012, 730, 1984,
         3157, 335, 3705, 3684, 3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792, 3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278,
         2296, 3833, 1343, 1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579, 1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533,
         1668, 1730, 1669, 3509, 2786])


def align_flame_to_multi_view_stereo(flame_template_mesh, multi_view_landmarks3d, reverse=False, kinect=False, mask_jaw=False):
    flame_template_landmarks = flame_template_mesh.vertices[FLAME_LANDMARK_INDICES, :]
    
    # remove NaNs
    mutliview_nans = np.any(np.isnan(multi_view_landmarks3d), axis=-1)
    valid = np.logical_not(mutliview_nans)
    
    if kinect:
        # if mask_jaw:
        #     # delete jaw points, added by jtang to address colinearility
        #     indices = np.arange(len(FLAME_LANDMARK_INDICES))
        #     valid = np.logical_and(valid, indices >=17)
        # else:
            # remove backgroud landmarks
            # backgroud = np.any(multi_view_landmarks3d>0.8, axis=-1)
            # nonbackgroud = np.logical_not(backgroud)
            # valid = np.logical_and(valid, nonbackgroud)

        # use the chink landmark
        indices = np.arange(len(FLAME_LANDMARK_INDICES))
        jaw = np.logical_and(indices>=7, indices<=9)
        selected = np.logical_or(jaw, indices>=17)
        valid = np.logical_and(valid, selected)
    else:
        # delete jaw points, added by jtang to address colinearility
        indices = np.arange(len(FLAME_LANDMARK_INDICES))
        valid = np.logical_and(valid, indices >=17)
        
    multi_view_landmarks3d = multi_view_landmarks3d[valid]
    flame_template_landmarks = flame_template_landmarks[valid]

    if reverse:
        # Compute the rigid transformation
        _, R, t = rigid_transform(multi_view_landmarks3d.T, flame_template_landmarks.T)

        # Filter outliers
        _multi_view_landmarks3d = multi_view_landmarks3d @ R.T + t
        dists = np.linalg.norm(_multi_view_landmarks3d - flame_template_landmarks, axis=-1)

        # non accurate initial alignment, so we delete it ???
        if not kinect:        
            valid = (dists < 0.02) 
            flame_template_landmarks = flame_template_landmarks[valid]
            multi_view_landmarks3d = multi_view_landmarks3d[valid]

        # Recompute, without the outliers
        s, R, t = similarity_transform(multi_view_landmarks3d, flame_template_landmarks)
    else:
        # Compute the rigid transformation
        _, R, t = rigid_transform(flame_template_landmarks.T, multi_view_landmarks3d.T)
        #Filter outliers
        _flame_template_landmarks = flame_template_landmarks @ R.T + t
        
        dists = np.linalg.norm(_flame_template_landmarks - multi_view_landmarks3d, axis=-1)

        if not kinect:         
            valid = (dists < 0.020 )
            flame_template_landmarks = flame_template_landmarks[valid]
            multi_view_landmarks3d = multi_view_landmarks3d[valid]

        # Recompute, without the outliers
        s, R, t = similarity_transform(flame_template_landmarks, multi_view_landmarks3d)

    # Visualize the result
    # flame_landmarks_transformed = s * flame_template_landmarks @ R.T + t
    # plotter = pv.Plotter()
    # plotter.add_points(multi_view_landmarks3d, color='red')
    # plotter.add_points(flame_landmarks_transformed, color='green')
    # for i in range(multi_view_landmarks3d.shape[0]):
    #     plotter.add_mesh(pv.Line(multi_view_landmarks3d[i, :], flame_landmarks_transformed[i, :]))
    # plotter.show()

    return s, R, t


'''
https://github.com/ClayFlannigan/icp/blob/master/icp.py
'''
from sklearn.neighbors import NearestNeighbors

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    #assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def align_flame_to_scans_icp(flame_template_mesh, points, normals, reverse=False, max_iterations=20, tolerance=0.001, normal_check=False, normal_threshold=0.2, distance_threshold=0.005): 
    if reverse:
        source_points, source_normals = points, normals
        target_points, target_normals = flame_template_mesh.vertices, flame_template_mesh.vertex_normals
    else:
        source_points, source_normals = flame_template_mesh.vertices, flame_template_mesh.vertex_normals
        target_points, target_normals = target_points, target_normals
    
    A, B = source_points, target_points
    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # transform opencv to opengl
    translation = src[:m, :].mean(axis=-1) #m
    rotation = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]], dtype='float32')
    translation_new = rotation @ translation
    init_pose = np.identity(m+1)
    init_pose[:m, :m] = rotation
    init_pose[:m, m] = translation_new
    
    # # apply the initial pose estimation
    src = np.dot(init_pose, src)
    T_final = init_pose 

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        #T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        _, R, t = rigid_transform(src[:m,:], dst[:m,indices])
        
        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        # updata T_final
        T_final = np.dot(T, T_final) 

        # update the current source
        src = np.dot(T, src)        

        # check error
        mean_error = np.mean(distances)
        #print('iter, {:d}, error {:.5f}'.format( i, mean_error))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    # T,_,_ = best_fit_transform(A, src[:m,:].T)
    # return T, distances, i

    # Filter outliers
    source_points_transformed = source_points @ T_final[:m, :m].T + T_final[:m, m]
    source_normals_transformed = source_normals @ T_final[:m, :m].T
    # if reverse:
    #     dists, indices = nearest_neighbor(target_points, source_points_transformed)
    #     valid = (dists < distance_threshold) 
        
    #     if normal_check:
    #         cosine_similarity = (source_normals_transformed[indices] * target_normals).sum(axis=-1)
    #         valid = valid &  (cosine_similarity < normal_threshold)

    #     # Recompute, without the outliers
    #     s, R, t = similarity_transform(source_points[indices][valid, :], target_points[valid, :])
    if True:
        #dists = np.linalg.norm(source_points_transformed - target_points, axis=-1)
        dists, indices = nearest_neighbor(source_points_transformed, target_points)
        valid = (dists < distance_threshold) 
        
        if normal_check:
            cosine_similarity = (source_normals_transformed * target_normals[indices, :]).sum(axis=-1)
            valid = valid &  (cosine_similarity < normal_threshold)

        # Recompute, without the outliers
        s, R, t = similarity_transform(source_points[valid, :], target_points[indices][valid, :])

    return s, R, t



def align_recon_to_scans_icp(points_recon, normals_recon, points, normals,  max_iterations=20, tolerance=0.001, normal_check=False, normal_threshold=0.2, distance_threshold=0.005): 

    source_points, source_normals = points_recon, normals_recon
    target_points, target_normals = points, normals
    
    A, B = source_points, target_points
    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # transform opencv to opengl
    translation = np.zeros((m,)) #src[:m, :].mean(axis=-1) #m
    rotation = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype='float32')
    translation_new = rotation @ translation
    init_pose = np.identity(m+1)
    init_pose[:m, :m] = rotation
    init_pose[:m, m] = translation_new
    
    # # apply the initial pose estimation
    src = np.dot(init_pose, src)
    T_final = init_pose 

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points 
        distances, indices = nearest_neighbor(dst[:m,:].T, src[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        _, R, t = rigid_transform(src[:m, indices], dst[:m, :])
        
        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        # updata T_final
        T_final = np.dot(T, T_final) 

        # update the current source
        src = np.dot(T, src)        

        # check error
        mean_error = np.mean(distances)
        #print('iter, {:d}, error {:.5f}'.format( i, mean_error))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    # T,_,_ = best_fit_transform(A, src[:m,:].T)
    # return T, distances, i

    # Filter outliers
    source_points_transformed = source_points @ T_final[:m, :m].T + T_final[:m, m]
    source_normals_transformed = source_normals @ T_final[:m, :m].T

    #dists = np.linalg.norm(source_points_transformed - target_points, axis=-1)
    dists, indices = nearest_neighbor(source_points_transformed, target_points)
    valid = (dists < distance_threshold) 
    
    if normal_check:
        cosine_similarity = (source_normals_transformed * target_normals[indices, :]).sum(axis=-1)
        valid = valid &  (cosine_similarity < normal_threshold)

    # Recompute, without the outliers
    _, R, t = rigid_transform(source_points[valid, :].T, target_points[indices][valid, :].T)

    return _, R, t







