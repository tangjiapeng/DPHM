from email.mime import image
import os
from plyfile import PlyElement, PlyData
import numpy as np
from PIL import Image
import open3d as o3d

def export_pointcloud(vertices, out_file, as_text=True):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)

def export_pointcloud_o3d(out_file, vertices, normals=None, colors=None, as_text=True):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices) # array_of_points.shape = (N,3)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)  # array_of_points.shape = (N,3)
    o3d.io.write_point_cloud(out_file, point_cloud)

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices

def load_pointcloud_normals(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    
    normals = np.stack([
        plydata['vertex']['nx'],
        plydata['vertex']['ny'],
        plydata['vertex']['nz']
    ], axis=1)
    
    return vertices, normals

def export_text(text_string, out_file):
    text_file = open(out_file, "w")
    text_file.write(text_string)
    text_file.close()

def export_image(image, out_file):
    image = (image * 255.0).astype(np.uint8)
    im = Image.fromarray(image)
    im.save(out_file)
    
def get_jet_color(v, vmin=0.0, vmax=1.0):
    """
    Maps
        map a vector clipped with the range [vmin, vmax] to colors
    Args:
        - vec (): 
    """ 
    c = np.array([1.0, 1.0, 1.0], dtype='float32')
    
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin

    if v < (vmin + 0.25 * dv):
        c[0] = 0
        c[1] = 4 * (v - vmin) / dv
    elif v < vmin + 0.5 * dv:
        c[0] = 0
        c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv
    elif v < vmin + 0.75 * dv:
        c[0] = 4 * (v - vmin - 0.5 * dv) / dv
        c[2] = 0
    else:
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[2] = 0
    return c

def vis_error_map(vertices, faces, error_npy, error_max=0.006):
    """
    Maps
        Convert a mesh with vertices errors to a colored mesh
        using Jet_color_map
    Args:
        - vertices: Nx3
        - faces:    Fx3
        - error_npy: N
    """ 
    error_min  = 0
    error_dist = error_max - error_min
    num_points  = error_npy.shape[0]
    error_map  = np.ones((num_points, 3), dtype='float32')

    mask = error_npy < error_min + 0.25 * error_dist
    error_map[mask, 0] = np.zeros((mask.sum()), dtype='float32')
    error_map[mask, 1] = 4 * (error_npy[mask] - error_min) / error_dist
    
    mask = (error_npy >= error_min + 0.25 * error_dist) & (error_npy < error_min + 0.5 * error_dist)
    error_map[mask, 0] = np.zeros((mask.sum()), dtype='float32')
    error_map[mask, 2] = 1 + 4 * (error_min + 0.25 * error_dist - error_npy[mask]) / error_dist
    
    mask = (error_npy >= error_min + 0.5 * error_dist) & (error_npy < error_min + 0.75 * error_dist)
    error_map[mask, 0] = 4 * (error_npy[mask] - error_min - 0.5 * error_dist) / error_dist
    error_map[mask, 2] = np.zeros((mask.sum()), dtype='float32')
    
    mask = error_npy >= error_min + 0.75 * error_dist
    error_map[mask, 1] = 1 + 4 * (error_min + 0.75 * error_dist - error_npy[mask]) / error_dist
    error_map[mask, 2] = np.zeros((mask.sum()), dtype='float32')
    
    # trimesh
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=error_map, process=False)
    # mesh.visual.vertex_colors = error_map
    
    # open3d
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(error_map)
    
    return mesh


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)