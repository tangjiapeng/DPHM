import os
import tyro
import numpy as np
import torch
import point_cloud_utils
import pyvista as pv
import pathlib
import trimesh
import math
import imageio
import tyro
from PIL import Image, ImageFilter

# Disable antialiasing:
import OpenGL.GL

suppress_multisampling = True
old_gl_enable = OpenGL.GL.glEnable

def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)

OpenGL.GL.glEnable = new_gl_enable

old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample

def new_glRenderbufferStorageMultisample(target, samples, internalformat, width, height):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(target, samples, internalformat, width, height)

OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample
import pyrender






KK = np.array([
    [2440, 0, 480],
    [0, 2440, 640],
    [0, 0, 1]], dtype=np.float32)

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(str(pathlib.Path(__file__).parent.resolve().absolute()) + "/shaders/mesh.vert",
                                                                 str(pathlib.Path(__file__).parent.resolve().absolute()) + "/shaders/mesh.frag",
                                                                 defines=defines)
        return self.program
    def clear(self):
        self.program = None

def render_glcam(model_in,  # model name or trimesh
                 K,
                 Rt,
                 rend_size=(512, 512),
                 znear=0.1,
                 zfar=2.0,
                 render_normals = True):

    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()

    if hasattr(mesh.visual, 'material'):
        mat = mesh.visual.material
        glossiness = mat.kwargs.get('Ns', 1.0)
        if isinstance(glossiness, list):
            glossiness = float(glossiness[0])
        roughness = (2 / (glossiness + 2)) ** (1.0 / 4.0)
        material = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            roughnessFactor=roughness,
            baseColorFactor=[255, 255, 255, 255],
            baseColorTexture=mat.image,
        )

        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material)
    else:
        glossiness = 1.0
        if isinstance(glossiness, list):
            glossiness = float(glossiness[0])
        roughness = (2 / (glossiness + 2)) ** (1.0 / 4.0)
        material = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            roughnessFactor=roughness,
            baseColorFactor=[255, 255, 255, 255],
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material)

    #pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material)

    # Scene creation
    scene = pyrender.Scene(ambient_light = [0.45,0.45,0.45, 1.0])

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)

    scene.add(cam, pose=Rt)

    # Set up the light
    instensity = 0.75

    light1 = pyrender.PointLight(intensity=instensity)
    light2 = pyrender.PointLight(intensity=instensity)

    light_pose1 = m3dLookAt(Rt[:3, 3]/2 + np.array([0, 0, 300]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))
    light_pose2 = m3dLookAt(Rt[:3, 3]/2 + np.array([0, 0, 0]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))

    light_pose1[:3, 3] = Rt[:3, 3]/2 + np.array([0.15, 0.1, -0.15])
    light_pose2[:3, 3] = Rt[:3, 3]/2 + np.array([-0.15, 0.1, -0.15])


    scene.add(light1, pose=light_pose1)
    scene.add(light2, pose=light_pose2)


    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   )
    if render_normals:
        r._renderer._program_cache = CustomShaderCache()

    normals_or_color, depth = r.render(scene,
                                       flags=pyrender.constants.RenderFlags.FLAT)
    #I = Image.fromarray(normals_or_color)
    #I.show()
    r.delete()
    if render_normals:
        world_space_normals = normals_or_color / 255 * 2 - 1

    depth[depth == 0] = float('inf')
    depth = (zfar + znear - (2.0 * znear * zfar) / depth) / (zfar - znear)

    if render_normals:
        return depth, world_space_normals
    else:
        return depth, normals_or_color


def get_3d_points(depth, K, Rt, rend_size=(512, 512), normals=None, znear=0.1, zfar=2.0):
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)


    xx, yy = np.meshgrid(np.arange(rend_size[0]), np.arange(rend_size[1]))
    xx = xx.reshape([-1])
    yy = yy.reshape([-1])
    pixel_inds = np.stack([xx, yy], axis=-1).astype(np.int32)
    lms3d = unproject_points(pixel_inds[:, :2], depth, rend_size, cam.get_projection_matrix(rend_size[1], rend_size[0]), Rt)

    return lms3d


def unproject_points(ppos, depth, rend_size, K, Rt):
    points = np.ones((ppos.shape[0], 4))
    points[:, [1, 0]] = ppos.astype(float)
    points[:, 0] = points[:, 0] / (rend_size[1] - 1) * 2 - 1
    points[:, 1] = points[:, 1] / (rend_size[0] - 1) * 2 - 1

    points[:, 1] *= -1
    ppos[:, 0] = np.clip(ppos[:, 0], 0, rend_size[0])
    ppos[:, 1] = np.clip(ppos[:, 1], 0, rend_size[1])
    points_depth = depth[ppos[:, 0], ppos[:, 1]]
    points[:, 2] = points_depth
    depth_cp = points[:, 2].copy()
    clipping_to_world = np.matmul(Rt, np.linalg.inv(K))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]
    points = points[:, :3]

    points[depth_cp >= 1, :] = np.NaN

    return points

def project_points(points, intrinsics, world_to_cam_pose):
    p_world = np.hstack([points, np.ones((points.shape[0], 1))])
    p_cam = p_world @ world_to_cam_pose.T
    depths = p_cam[:, [2]]  # Assuming OpenCV convention: depth is positive z-axis
    p_cam = p_cam / depths
    p_screen = p_cam[:, :3] @ intrinsics.T
    p_screen[:, 2] = np.squeeze(depths, 1)  # Return depth as third coordinate
    return p_screen

def project_points_torch(points, intrinsics, world_to_cam_pose):
    p_world = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=-1)
    p_cam = p_world @ world_to_cam_pose.T
    depths = p_cam[:, [2]]  # Assuming OpenCV convention: depth is positive z-axis
    p_cam = p_cam / depths
    p_cam[:, 0] *= -1 # since the NeuS rendering is happening in OpenGL we need to correct it as such
    p_screen = p_cam[:, :3] @ intrinsics.T
    p_screen[:, 2] = depths.squeeze(dim=1)  # Return depth as third coordinate
    return p_screen

def m3dLookAt(eye, target, up):
    mz = (eye-target)
    mz /= np.linalg.norm(mz, keepdims=True)  # inverse line of sight
    mx = np.cross(up, mz)
    mx /= np.linalg.norm(mx, keepdims=True)
    my = np.cross(mz, mx)
    my /= np.linalg.norm(my)
    tx = eye[0] #np.dot(mx, eye)
    ty = eye[1] #np.dot(my, eye)
    tz = eye[2] #-np.dot(mz, eye)
    return np.array([[mx[0], my[0], mz[0], tx],
                     [mx[1], my[1], mz[1], ty],
                     [mx[2], my[2], mz[2], tz],
                     [0, 0, 0, 1]])


def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def render_and_backproject(m,
                           down_scale_factor : int = 2,
                           crop : int = 50):
    E = m3dLookAt(np.array([0, 0, 1]) * 0.6,
                  np.zeros([3]),
                  np.array([0, 1, 0]))

    rend_size = (1280 // down_scale_factor, 960 // down_scale_factor)
    crop = crop // down_scale_factor

    KK = np.array(
        [[2440 / down_scale_factor, 0.00000000e+00, (rend_size[1] / 2) - crop],
         [0.00000000e+00, 2440 / down_scale_factor, (rend_size[0] / 2) - crop],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
    )
    rend_size = (rend_size[0] - 2 * crop, rend_size[1] - 2 * crop)

    m.vertices /= 4
    depth, rgb = render_glcam(m, KK, E, rend_size=rend_size, render_normals=False)
    points3d = get_3d_points(depth, KK, E, rend_size=rend_size)
    points3d *= 4
    m.vertices *= 4

    #valid = np.logical_not(np.any(np.isnan(points3d), axis=-1))
    #points3d = points3d[valid, :]

    #normals = np.transpose(normals, [1, 0, 2])
    #normals = normals.reshape([-1, 3])
    #normals = normals[valid, :]
    return rgb, points3d

def gen_render_samples(m, N, scale=4, down_scale_factor=1, crop=0, render_color=False, return_grid=False):
    m = m.copy()
    m.vertices /= scale

    rend_size = (1280 // down_scale_factor, 960 // down_scale_factor)
    crop = crop // down_scale_factor

    KK = np.array(
        [[2440 / down_scale_factor, 0.00000000e+00, (rend_size[1] / 2) - crop],
         [0.00000000e+00, 2440 / down_scale_factor, (rend_size[0] / 2) - crop],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
    )
    rend_size = (rend_size[0] - 2 * crop, rend_size[1] - 2 * crop)




    cams = fibonacci_sphere(N + 2)[1:-1]
    cams.reverse()
    if render_color or return_grid:
        eyes = np.array(fibonacci_sphere(1000))
        eyes = eyes[eyes[:, 2] > 0.5, :]
        eyes = eyes[eyes[:, 1] < 0.7, :]
        eyes = eyes[eyes[:, 1] > -0.7, :]
        #eyes = eyes[eyes[:, 2] > 0.6, :]
        #eyes = eyes[eyes[:, 1] < 0.55, :]
        #eyes = eyes[eyes[:, 1] > -0.55, :]
        cams = []
        for i in range(N):
            if i == 0:
                cams.append(np.array([0, 0, 1]))
            else:
                rnd_indx = np.random.randint(0, len(eyes))
                #rnd_indx = 10
                cams.append(eyes[rnd_indx])

    all_points = []
    all_normals = []

    for cam_origin in cams:

        if N == 1:
            cam_origin = [0, 0, 1]
        E = m3dLookAt(np.array(cam_origin) * 0.6,
                      np.zeros([3]),
                      np.array([0, 1, 0]))

        depth, normals = render_glcam(m, KK, E, rend_size=rend_size, render_normals=not render_color)
        if not render_color:
            n = normals + 1
            n /= 2.0
            n *= 255.0
            n = n.astype(np.uint8)
            #I = Image.fromarray(n)
            #I.show()
        points3d = get_3d_points(depth, KK, E, rend_size=rend_size)

        if not render_color and not return_grid:
            valid = np.logical_not(np.any(np.isnan(points3d), axis=-1))
            points3d = points3d[valid, :]

            normals = np.transpose(normals, [1, 0, 2])
            normals = normals.reshape([-1, 3])
            normals = normals[valid, :]
            # back face removal
            ray_dir = points3d - np.array(cam_origin) * 0.6
            ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
            angle = np.sum(ray_dir * normals, axis=-1)

            all_points.append(points3d[angle < -0.01, :])
            all_normals.append(normals[angle < -0.01, :])
        else:
            if not render_color:

                ray_dir = points3d - np.array(cam_origin) * 0.6
                ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
                ray_dir = ray_dir.reshape([normals.shape[1], normals.shape[0], 3])
                ray_dir = np.transpose(ray_dir, [1, 0, 2])

                #point_img = ray_dir
                #point_img -= np.nanmin(point_img)
                #point_img /= np.nanmax(point_img)
                #point_img *= 255
                #point_img = point_img.astype(np.uint8)
                #point_img[np.isnan(point_img)] = 255
                #I = Image.fromarray(point_img)
                #I.show()
#
                #point_img = normals
                #point_img -= np.nanmin(point_img)
                #point_img /= np.nanmax(point_img)
                #point_img *= 255
                #point_img = point_img.astype(np.uint8)
                #point_img[np.isnan(point_img)] = 255
                #I = Image.fromarray(point_img)
                #I.show()
#
#
                angle = np.sum(ray_dir * normals, axis=-1)
                #point_img = angle
                #point_img -= np.nanmin(point_img)
                #point_img /= np.nanmax(point_img)
                #point_img *= 255
                #point_img = point_img.astype(np.uint8)
                #point_img[np.isnan(point_img)] = 255
                #I = Image.fromarray(point_img)
                #I.show()


                valid_angle = angle < -0.01
                valid_angle[np.isnan(angle)] = False
                #valid_angle = valid_angle.reshape([normals.shape[1], normals.shape[0], 3])
                #valid_angle = np.transpose(valid_angle, [1, 0, 2])




            point_img = points3d.reshape([normals.shape[1], normals.shape[0], 3])
            point_img = np.transpose(point_img, [1, 0, 2])

            #if not render_color:
            #    point_img[np.logical_not(valid_angle)] = np.NaN
            #    normals[np.logical_not(valid_angle)] = np.NaN

            #point_img -= np.nanmin(point_img)
            #point_img /= np.nanmax(point_img)
            #point_img *= 255
            #point_img = point_img.astype(np.uint8)
            #point_img[np.isnan(point_img)] = 255
            #I = Image.fromarray(point_img)
            #I.show()
            #In = Image.fromarray(normals)
            #In.show()

            #p = np.reshape(point_img, [-1, 3])
            #pl = pv.Plotter()
            #pl.add_points(p)
            #pl.show()



            all_points.append(point_img)
            all_normals.append(normals)

    if render_color or return_grid:
        return [points*scale for points in all_points],\
                all_normals
    else:
        return np.concatenate(all_points, axis=0)*scale,\
               np.concatenate(all_normals, axis=0)


class SimpleMesh():
    def __init__(self, v, f):
        self.v = v
        self.f = f

def main(start_s: int, stop_s : int):

    from time import time
    print('STARTING SCRIPT WITH {} {}'.format(start_s, stop_s))
    import distinctipy

    #test_dir = '/home/giebenhain/test_vgg_fine/'
    #os.makedirs(test_dir, exist_ok=True)
    from GTA.utils.mesh_operations import cut_trimesh_vertex_mask
    from GTA.data.manager import DataManager
    import pyvista as pv

    manager = DataManager()
    N = 20
    subject = 293
    expression = 0

    for subject in manager.get_all_subjects()[start_s:stop_s]:
        for expression in manager.get_expressions(subject):
            #if True:
            try:
                print(subject, expression)

                t0 = time()
                m = manager.get_raw_mesh(subject, expression)

                #m.show()




                mesh = manager.get_raw_mesh(subject, expression, mesh_type='pcu', textured=True)
                t1 = time()
                print(f'Loading meshes {t1 - t0}')
                t0 = t1

                #mesh = cut_trimesh_vertex_mask(mesh, valid)
                #mesh = point_cloud_utils.TriangleMesh()
                #mesh.vertex_data.positions = m.vertices
                #mesh.face_data.vertex_ids = m.faces
                #mesh = SimpleMesh(m.vertices, m.faces)
                valid_face_inds = remove_large_triangles(mesh, max_size_factor=10)

                #m.show()

                #m2 = trimesh.Trimesh(m.vertices, m.faces[valid_face_inds], process=False)
                m.faces = m.faces[valid_face_inds]
                #m.show()

                valid = manager.cut_throat(m.vertices, subject, expression)
                m = cut_trimesh_vertex_mask(m, valid)

                t1 = time()
                print(f'Cut Mesh {t1 - t0}')
                t0 = t1





                if False:
                    #m.show()
                    rgb, points3d = render_and_backproject(m)

                    I = Image.fromarray(rgb)
                    I.show()

                    pl = pv.Plotter()
                    pl.add_mesh(m)
                    pl.add_points(points3d)
                    pl.show()

                N = 2
                all_points, all_normals = gen_render_samples(m, N, scale=4, render_color=True, down_scale_factor=1) # points + color
                all_points2, all_actual_normals = gen_render_samples(m, N, scale=4, render_color=False, down_scale_factor=1, return_grid=True) # same points + normals

                t1 = time()
                print(f'Rendered {t1 - t0}')
                t0 = t1


                SAVE_DIR = f'/cluster/andram/sgiebenhain/training_data_nphm/rendered_images/{subject:03d}/{expression:03d}/'
                if os.path.exists('/mnt/rohan'):
                    SAVE_DIR = '/mnt/rohan' + SAVE_DIR
                os.makedirs(SAVE_DIR, exist_ok=True)

                for i in range(len(all_points)):
                    p3d = all_points[i].astype(np.float16)
                    n3d = all_actual_normals[i].astype(np.float16)

                    #p3d_f16 = p3d.copy().astype(np.float16)
                    #stacked = np.stack([p3d, n3d], axis=-1).astype(np.float16)

                    #imageio.imwrite(f'{SAVE_DIR}/points_{i:03d}.exr', p3d)
                    #p3d_exr = imageio.imread(f'{SAVE_DIR}/points_{i:03d}.exr')



                    rgb = Image.fromarray(all_normals[i])
                    rgb.save(f'{SAVE_DIR}/rgb_{i:03d}.png')
                    np.savez_compressed(f'{SAVE_DIR}/points_{i:03d}.npz', p3d)
                    #np.savez_compressed(f'{SAVE_DIR}/geo_{i:03d}_f16.npz', stacked)
                    np.savez_compressed(f'{SAVE_DIR}/normals_{i:03d}.npz', n3d)

                    #pl = pv.Plotter()
                    #pl.add_points(p3d.reshape(-1, 3))
                    #pl.add_points(p3d_f16.astype(np.float32).reshape(-1, 3), color='red')
                    ##pl.add_points(p3d_exr.reshape(-1, 3)[mask, :], color='green')
                    #pl.show()
                    t1 = time()
                    print(f'Saved {t1 - t0}')
                    t0 = t1
            except Exception as e:
                pass


    exit()




    #pl = pv.Plotter()
    #pl.add_points(all_points2[0].reshape(-1, 3), scalars=all_actual_normals[0][:, :, 0].reshape(-1 ))
    #pl.add_points(all_points[0].reshape(-1, 3), scalars=all_normals[0][:, :, 0].reshape(-1 ))
    #pl.add_mesh(m)
    #pl.show()
#
    #point_img = all_actual_normals[0]
    #point_img -= np.nanmin(point_img)
    #point_img /= np.nanmax(point_img)
    #point_img *= 255
    #point_img = point_img.astype(np.uint8)
    #point_img[np.isnan(point_img)] = 255
    #I = Image.fromarray(point_img)
    #I.show()



    for i in range(len(all_actual_normals)):
        invalid = np.any(np.isnan(all_actual_normals[i]), axis=-1)

        # dilate white region by one pixel and set points etc to NaN accordingly!
        white_region = Image.fromarray(np.all(all_normals[i] == 255, axis=-1))
        #white_region.show()
        dilation_img = white_region.filter(ImageFilter.MaxFilter(11))
        #dilation_img.show()

        invalid2 = np.array(dilation_img)
        invalid = invalid2 | invalid

        all_points[i][invalid, :] = np.NaN
        #tmp = all_normals[i].copy()
        #tmp[invalid, :] = 255
        #all_normals[i] = tmp


    total_img = 0
    for img, points in zip(all_normals, all_points):

        #pl = pv.Plotter()
        #pl.add_points(np.reshape(points, [-1, 3]), scalars=np.reshape(img, [-1, 3]), rgb=True)
        #pl.show()

        I = Image.fromarray(img)
        I.show()
        continue


        assert 1 == 2
        crop_size = 64
        num_samples = 8
        crops, points_crops = autocrop(I, points=points,
                         pct_focus=0., matrix=crop_size, sample=num_samples**2, down_scale_factor=2, max_pct_white=0.15)
        #crops, points_crops = autocrop(I, points=points,
        #                 pct_focus=0., matrix=crop_size, sample=num_samples**2, down_scale_factor=4, max_pct_white=0.15)
        big_img = np.zeros([crop_size*num_samples, crop_size*num_samples, 3], dtype=np.uint8)
        for i, (crop, crop_points) in enumerate(zip(crops, points_crops)):
            #if i == 0:
            #    crop.show()
            crop.save(test_dir + '{}.png'.format(total_img))
            np.save(test_dir + '{}.npy'.format(total_img), np.concatenate([crop_points, crop], axis=-1))
            #pl = pv.Plotter()
            #pl.add_points(np.reshape(crop_points, [-1, 3]), scalars=np.reshape(crop, [-1, 3]), rgb=True)
            #pl.show()
            #pl.add_points
            total_img += 1
            big_img[i//num_samples*crop_size:(i//num_samples + 1)*crop_size, i%num_samples * crop_size: (i%num_samples+1)*crop_size, :] = np.array(crop)
        bigI = Image.fromarray(big_img)
        bigI.show()
    assert 1==2
    colors = distinctipy.get_colors(N)
    pl = pv.Plotter()
    pl.add_mesh(m)
    for i, p3d in enumerate(all_points):
         pl.add_points(p3d, color=colors[i])
    #pl.add_points(all_points, scalars=all_normals[:, 0])
    #    print(p3d.shape)
    pl.show()


if __name__ == '__main__':
    tyro.cli(main)



