import faiss
import torch
import trimesh
import numpy as np
import pyvista as pv
import torch.nn as nn
import os
import math
import networkx as nx
from dphm_tum.flame.flame_model import FLAME
from dphm_tum.utils.flame_utils import euler_angles_to_matrix
from pytorch3d.transforms import quaternion_to_matrix
# TODO: Important to enable torch support for faiss
import faiss.contrib.torch_utils
faiss_resource_manager = faiss.StandardGpuResources()
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

def enlarge_region(idx, one_ring):
    res = []
    for j in range(len(idx)):
        res += one_ring[idx[j]]
    res = np.unique(np.array(res))
    return res

def random_sample(point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm, num_points=100000):
    rnd_ind = torch.randint(0, point_cloud_np.shape[0], size=[num_points])
    point_cloud_np = point_cloud_np[rnd_ind, :]
    normals_np = normals_np[rnd_ind, :]
    point_cloud_np_nphm = point_cloud_np_nphm[rnd_ind, :]
    normals_np_nphm = normals_np_nphm[rnd_ind, :]
    return point_cloud_np, normals_np, point_cloud_np_nphm, normals_np_nphm


def add_gaussian_noise(point_cloud_np, normals_np, noise_level=0.0):
    point_cloud_np_noise = point_cloud_np + torch.from_numpy(np.random.normal(loc=0.0, scale=noise_level, size=(point_cloud_np.shape[0], point_cloud_np.shape[1]))).float().to(point_cloud_np.device)
    normals_np_noise = normals_np + torch.from_numpy(np.random.normal(loc=0.0, scale=noise_level, size=(normals_np.shape[0], normals_np.shape[1]))).float().to(normals_np.device)
    normals_np_noise = normals_np_noise / torch.norm(normals_np_noise, dim=1, keepdim=True) 
    return point_cloud_np_noise, normals_np_noise


class DeformationFLAME(nn.Module):
    def __init__(self, config, all_point3d, all_normals3d, all_landmarks3d, flame_parameters, _device='cuda:0', 
                 use_normals=True, use_landmark=True, use_jaw=True, mask_forehead=False, use_quanternion=False, euler_convention='XYZ', from_flame_to_scan=True,
                  sample=False, num_points=100000, noise_level=0):
        super(DeformationFLAME, self).__init__()
        self.z_shape_flame = config.z_shape_flame
        self.z_expression_params = config.z_expression_flame
        self.num_flame_vertices = config.num_flame_vertices
        self.flame_segmented_mesh_path = config.flame_segmented_mesh_path
        self.device = _device
        self.FLAME = FLAME(config).to(_device)
        self.normal_threshold = 0.2
        self.distance_threshold = 0.005

        self.config = config
        self.use_normals = use_normals
        self.use_landmark = use_landmark
        self.use_jaw = use_jaw
        self.mask_foreface = mask_forehead
        self.use_quanternion = use_quanternion
        self.euler_convention = euler_convention
        self.from_flame_to_scan = from_flame_to_scan
        
        self.sample = sample
        self.num_points = num_points
        self.noise_level = noise_level

        # Setup mask
        self.setup_vertex_mask()
        print('set up vertex mask')

        # Get the warped paarmeters
        # warped_vertices_list, _ = self.warp(flame_parameters) # why not apply transform ?
        warped_vertices_list, _ = self.warp(flame_parameters, transform=True)

        # Initialize Temporal Flame Mesh Parameters
        self.template_frames = [trimesh.Trimesh(v.squeeze().detach().cpu().numpy(), self.FLAME.faces_tensor.cpu().numpy(), process=False) for v in warped_vertices_list]
        self.template_vertex_normals_frames = torch.stack([torch.from_numpy(np.array(t.vertex_normals)).float() for t in self.template_frames], dim=0).to(_device)
        self.template_face_normals_frames = torch.stack([torch.from_numpy(np.array(t.face_normals)).float() for t in self.template_frames], dim=0).to(_device)
        self.template_mesh_faces = torch.from_numpy(np.array(self.template_frames[0].faces)).to(_device)

        min_len_point3d = all_point3d[0].shape[0]
        for i in range(1, len(all_point3d)):
            if min_len_point3d > all_point3d[i].shape[0]:
                min_len_point3d = all_point3d[i].shape[0]
        self.all_point3d = torch.stack([torch.from_numpy(points3d[:min_len_point3d, :]) for points3d in all_point3d], dim=0).to(self.device).float()
        self.all_normals3d = torch.stack([torch.from_numpy(normals3d[:min_len_point3d, :]) for normals3d in all_normals3d], dim=0).to(self.device).float()
        self.all_landmarks3d = torch.stack([torch.from_numpy(np.array(t)).float() for t in all_landmarks3d], dim=0).to(self.device).float()
        print('load all observed data')
        
        # Setup Faiss Index
        if False:
            self.setup_faiss_index()
        else:
            self.indices = [_ for i in range(len(self.all_point3d))]
            self.indices_matches = [torch.zeros(5023, 1, device=self.device, dtype=torch.int64) for i in range(len(self.all_point3d))]
        print('initial DeformationFLAME')

    def setup_vertex_mask(self):

        # TODO: Template mesh with color encodings
        mesh_colored = trimesh.load(self.flame_segmented_mesh_path, process=False)
        vertex_colors = mesh_colored.visual.vertex_colors
        # Masking inner lip region
        inner_lip_mask = np.logical_and(np.logical_and(vertex_colors[:, 0] == 0, vertex_colors[:, 1] == 0), vertex_colors[:, 2] == 255)
        inner_lip_mask = np.logical_and(inner_lip_mask, mesh_colored.vertices[:, 2] < 0.0)
        inner_lip_idx = np.where(inner_lip_mask)[0]

        # Enlarge region by including k-hop neighborhood
        graph = nx.from_edgelist(mesh_colored.edges_unique)
        one_ring = [list(graph[i].keys()) for i in range(len(mesh_colored.vertices))]
        enlarged_idx = inner_lip_idx.copy()
        for i in range(3):
            enlarged_idx = enlarge_region(enlarged_idx.copy(), one_ring).copy()
        template_mask = np.ones([mesh_colored.vertices.shape[0]], dtype=bool)
        template_mask[enlarged_idx] = 0

        # Mask out back of the head/hair region
        hair_region = np.logical_and(np.logical_and(vertex_colors[:, 0] == 255, vertex_colors[:, 1] == 0), vertex_colors[:, 2] == 0)
        template_mask[hair_region] = 0
        self.template_mask = torch.from_numpy(template_mask).to(self.device)

    def setup_faiss_index(self):
        faiss_handler = faiss.GpuClonerOptions()

        self.quantizers = []
        self.indices = []
        for idx in range(len(self.all_point3d)):
            quantizer = faiss.IndexFlatL2(3)
            faiss_metric = faiss.METRIC_L2
            index = faiss.IndexIVFFlat(quantizer, 3, 1000, faiss_metric)
            index = faiss.index_cpu_to_gpu(faiss_resource_manager, 0, index, faiss_handler)
            index.train(self.all_point3d[idx])
            index.nprobe = 10
            self.quantizers.append(quantizer)
            self.indices.append(index)

        self.distances = [torch.zeros(5023, 1, device=self.device, dtype=torch.float32) for i in range(len(self.all_point3d))]
        self.indices_matches = [torch.zeros(5023, 1, device=self.device, dtype=torch.int64) for i in range(len(self.all_point3d))]

        for idx, index in enumerate(self.indices):
            index.train(self.all_point3d[idx])
            index.add(self.all_point3d[idx])
            index.probe = 10
    
    def rigid_transform_from_flame_to_scan(self, points, flame_parameters, idx=None, normals=None, batch=False):
        if self.use_quanternion:
            global_rotation = quaternion_to_matrix(flame_parameters['rotation'])
        else:
            global_rotation = euler_angles_to_matrix(flame_parameters['rotation'], self.euler_convention) # B X 3 x 3
        
        global_translation = flame_parameters['translation']  # B X 3
        global_scale = flame_parameters['scale']  # B X 1
        if self.from_flame_to_scan:
            if batch:
                points_transform = global_scale * torch.einsum('bxy,bny->bnx', global_rotation,  points) + global_translation.unsqueeze(1) 
                normals_transform  = torch.einsum('bxy,bny->bnx', global_rotation,  normals) if normals is not None else None
            else:
                points_transform = global_scale * torch.einsum('xy,ny->nx', global_rotation[idx],  points) + global_translation[idx].unsqueeze(0) 
                normals_transform  = torch.einsum('xy,ny->nx', global_rotation[idx],  normals) if normals is not None else None
        else:
            # R.inverse == R.T
            if batch:
                points_transform = (1 / global_scale) * torch.einsum('bxy,bny->bnx', global_rotation.permute(0, 2, 1), (points - global_translation.unsqueeze(1)) ) 
                normals_transform  = torch.einsum('bxy,bny->bnx', global_rotation.permute(0, 2, 1), normals) if normals is not None else None
            else:
                points_transform = (1 / global_scale) * torch.einsum('xy,ny->nx', global_rotation[idx].T, (points - global_translation[idx].unsqueeze(0)) ) 
                normals_transform  = torch.einsum('xy,ny->nx', global_rotation[idx].T, normals) if normals is not None else None
        return points_transform, normals_transform
        
    def rigid_transform_from_scan_to_flame(self, points, flame_parameters, idx=None, normals=None, batch=False):
        if self.use_quanternion:
            global_rotation = quaternion_to_matrix(flame_parameters['rotation'])
        else:
            global_rotation = euler_angles_to_matrix(flame_parameters['rotation'], self.euler_convention) # B X 3 x 3
        global_translation = flame_parameters['translation']  # B X 3
        global_scale = flame_parameters['scale']  # B X 1
        if self.from_flame_to_scan:
            # R.inverse == R.T
            if batch:
                points_transform = (1 / global_scale) * torch.einsum('bxy,bny->bnx', global_rotation.permute(0, 2, 1), (points - global_translation.unsqueeze(1)) ) 
                normals_transform  = torch.einsum('bxy,bny->bnx', global_rotation.permute(0, 2, 1), normals) if normals is not None else None
            else:
                points_transform = (1 / global_scale) * torch.einsum('xy,ny->nx', global_rotation[idx].T, (points - global_translation[idx].unsqueeze(0)) ) 
                normals_transform  = torch.einsum('xy,ny->nx', global_rotation[idx].T, normals) if normals is not None else None
        else:
            if batch:
                points_transform = global_scale * torch.einsum('bxy,bny->bnx', global_rotation,  points) + global_translation.unsqueeze(1) 
                normals_transform  = torch.einsum('bxy,bny->bnx', global_rotation,  normals) if normals is not None else None
            else:
                points_transform = global_scale * torch.einsum('xy,ny->nx', global_rotation[idx],  points) + global_translation[idx].unsqueeze(0) 
                normals_transform  = torch.einsum('xy,ny->nx', global_rotation[idx],  normals) if normals is not None else None
        return points_transform, normals_transform

    def mask_foreface_region(self, flame_parameters, idx):
        warped_point3d, _ = self.rigid_transform_from_scan_to_flame(self.all_point3d[idx], flame_parameters, idx, normals=None)    
        obs = warped_point3d.clone()
        
        #valid = (obs[:, 1] > -0.4/4 )  & (obs[:, 2] > 0.0/4)  &  (obs[:, 2] < 0.5/4)
        
        # valid = (obs[:, 1] > -0.5/4 ) & (obs[:, 2] > -0.2/4) &  ( obs[:, 2] < 0.6/4) & (obs[:, 0] > -0.55/4) &  (obs[:, 0] < 0.55/4) 
        
        valid = (obs[:, 1] > -0.55/4 ) & (obs[:, 2] > -0.4/4) &  ( obs[:, 2] < 0.6/4) & (obs[:, 0] > -0.55/4) &  (obs[:, 0] < 0.55/4) 
        
        return valid

    def warp(self, flame_parameters, transform=True):
        jaw_pose = flame_parameters['jaw_pose']
        num_frames = flame_parameters['expression'].shape[0]
        vertices, landmarks = self.FLAME(flame_parameters['shape'].repeat(num_frames, 1), flame_parameters['expression'],
                                         torch.cat([torch.zeros_like(jaw_pose), jaw_pose], dim=-1))  # Head Pose, Jaw Pose

        # Important
        if transform:
            # Apply head pose changes from flame to scan space
            warped_vertices, _  =  self.rigid_transform_from_flame_to_scan(vertices, flame_parameters, batch=True)
            warped_landmarks, _ =  self.rigid_transform_from_flame_to_scan(landmarks, flame_parameters, batch=True)
        else:
            warped_vertices = vertices
            warped_landmarks = landmarks
        return warped_vertices, warped_landmarks

    def get_correspondences(self, warped_vertices, flame_parameters):

        all_point3d_matches = []
        all_normals3d_matches = []
        all_invalid_indices = []
        all_normals_scale = []
        for idx in range(len(self.indices)):
            if False:
                self.distances[idx], self.indices_matches[idx] = self.indices[idx].search(warped_vertices[idx, ...].float().contiguous(), 1, self.distances[idx],
                                                                                        self.indices_matches[idx])
            else:
                # mask points that are out of regions
                if self.mask_foreface:
                    valid_idx = self.mask_foreface_region(flame_parameters, idx)
                    points_valid = self.all_point3d[idx, valid_idx]
                    normals_valid = self.all_normals3d[idx, valid_idx]
                else:
                    points_valid = self.all_point3d[idx, :]
                    normals_valid = self.all_normals3d[idx, :]
                
                if self.sample:
                    points_valid = points_valid[:self.num_points]
                    normals_valid = normals_valid[:self.num_points]
                    points_valid, normals_valid = add_gaussian_noise(points_valid, normals_valid, noise_level=self.noise_level)
                    
                # apply possible transformation to scans
                # fix points always at camera space
                #points_valid, normals_valid = self.rigid_transform_from_scan_to_flame(points_valid, flame_parameters, idx, normals=normals_valid)
                # for each vert, find nn point from scan
                nns = knn_points(warped_vertices[idx:idx+1], points_valid.unsqueeze(0), K=1, return_nn=True, norm=1)
                self.indices_matches[idx] = nns.idx.squeeze()
            points3d_matches = points_valid[self.indices_matches[idx].squeeze(), :]
            normals3d_matches = normals_valid[self.indices_matches[idx].squeeze(), :]
            normal_scale = (self.template_vertex_normals_frames[idx] * normals3d_matches).sum(dim=-1)
            distance = (points3d_matches - warped_vertices[idx, ...]).norm(dim=-1)
            invalid_distance = distance > self.distance_threshold
            if self.use_normals:
                invalid_normals = normal_scale < self.normal_threshold
                total_invalid_indices = torch.logical_or(invalid_normals, invalid_distance)
            else:
                total_invalid_indices = invalid_distance

            all_point3d_matches.append(points3d_matches)
            all_normals3d_matches.append(normals3d_matches)
            all_invalid_indices.append(total_invalid_indices)
            all_normals_scale.append(normal_scale)
        return torch.stack(all_point3d_matches, dim=0), torch.stack(all_normals3d_matches, dim=0), torch.stack(all_invalid_indices, dim=0), torch.stack(all_normals_scale, dim=0)
    
    
    def get_correspondences_scan2mesh(self, warped_vertices, flame_parameters):

        all_point3d_matches = []
        all_normals3d_matches = []
        all_invalid_indices = []
        all_normals_scale = []
        all_vertices_matches = []
        all_vertices_normals_matches = []
        for idx in range(len(self.indices)):
            # mask points that are out of regions
            if self.mask_foreface:
                valid_idx = self.mask_foreface_region(flame_parameters, idx)
                points_valid = self.all_point3d[idx, valid_idx]
                normals_valid = self.all_normals3d[idx, valid_idx]
            else:
                points_valid = self.all_point3d[idx, :]
                normals_valid = self.all_normals3d[idx, :]
            
            # downsample
            num_samples = 30000
            sample_idx = torch.randint(0, points_valid.shape[0], (num_samples,)).to(points_valid.device)
            points_valid = points_valid[sample_idx, :]
            normals_valid = normals_valid[sample_idx, :]      
                
            # for each vert, find nn point from scan
            nns = knn_points(points_valid.unsqueeze(0), warped_vertices[idx:idx+1], K=1, return_nn=True, norm=1)
            self.indices_matches[idx] = nns.idx.squeeze()
            
            warped_vertices_matches = warped_vertices[idx][self.indices_matches[idx].squeeze(), :]
            warped_vertices_normals_matches = self.template_vertex_normals_frames[idx][self.indices_matches[idx].squeeze(), :]
            distance = (points_valid - warped_vertices_matches).norm(dim=-1)
            invalid_distance = distance > self.distance_threshold
            normal_scale = (normals_valid * warped_vertices_normals_matches).sum(dim=-1)
            if self.use_normals:
                invalid_normals = normal_scale < self.normal_threshold
                total_invalid_indices = torch.logical_or(invalid_normals, invalid_distance)
            else:
                total_invalid_indices = invalid_distance

            all_point3d_matches.append(points_valid)
            all_normals3d_matches.append(normals_valid)
            all_invalid_indices.append(total_invalid_indices)
            all_normals_scale.append(normal_scale)
            all_vertices_matches.append(warped_vertices_matches)
            all_vertices_normals_matches.append(warped_vertices_normals_matches)
        return torch.stack(all_point3d_matches, dim=0), torch.stack(all_normals3d_matches, dim=0), \
            torch.stack(all_invalid_indices, dim=0), torch.stack(all_normals_scale, dim=0), \
            torch.stack(all_vertices_matches, dim=0), torch.stack(all_vertices_normals_matches, dim=0),
            


    def compute_landmark_residual(self, warped_landmark_list, flame_parameters):        
        # Landmark loss
        total_landmark_residual = 0
        # Landmarks computed on FLAME mesh and warped back to MVS Space (warped_landmark_list) should match the landmarks from Multi View Stereo (all_landmarks3d)
        residuals = (warped_landmark_list - self.all_landmarks3d).abs()

        # Jaw weights
        if self.use_jaw:
            jaw_idx = torch.arange(17, device=self.device)
            residuals[:, jaw_idx, :] *= 0.01
        else:
            jaw_idx = torch.arange(17, device=self.device)
            residuals[:, jaw_idx, :] = 0
            
        # Eye weights
        eye_idx = torch.arange(36, 48, device=self.device)
        residuals[:, eye_idx, :] *= 32

        # Mouth weights
        mouth_idx = torch.arange(48, 68, device=self.device)
        residuals[:, mouth_idx, :] *= 32

        non_nan_indices = ~torch.isnan(residuals)
        residuals = torch.where(non_nan_indices, residuals, torch.tensor(0.0, device=self.device, dtype=torch.float32))
        total_landmark_residual += residuals.sum()
        total_landmark_residual /= residuals.shape[0]
        return total_landmark_residual
    
    def compute_landmark2d_residual(self, warped_landmark_list, flame_parameters):        
        # Landmark loss
        total_landmark_residual = 0
        # Landmarks computed on FLAME mesh and warped back to MVS Space (warped_landmark_list) should match the landmarks from Multi View Stereo (all_landmarks3d)
        warped_landmark_2d_list = warped_landmark_list[:, :, 0:2] / warped_landmark_list[:, :, 2:3]
        all_landmarks2d =   self.all_landmarks3d[:, :, 0:2] / self.all_landmarks3d[:, :, 2:3]
        residuals = (warped_landmark_2d_list - all_landmarks2d).abs()

        # Jaw weights
        #if self.use_jaw:
        jaw_idx = torch.arange(17, device=self.device)
        residuals[:, jaw_idx, :] *= 32
        #    residuals[:, jaw_idx, :] *= 0.01
        #else:
            # jaw_idx = torch.arange(17, device=self.device)
            # residuals[:, jaw_idx, :] = 0
            
        # Eye weights
        eye_idx = torch.arange(36, 48, device=self.device)
        residuals[:, eye_idx, :] *= 32

        # Mouth weights
        mouth_idx = torch.arange(48, 68, device=self.device)
        residuals[:, mouth_idx, :] *= 32

        non_nan_indices = ~torch.isnan(residuals)
        residuals = torch.where(non_nan_indices, residuals, torch.tensor(0.0, device=self.device, dtype=torch.float32))
        total_landmark_residual += residuals.sum()
        total_landmark_residual /= residuals.shape[0]
        return total_landmark_residual

    def compute_geometric_residual(self, warped_vertices, flame_parameters):
        # Geometric loss
        num_frames = warped_vertices.shape[0]
        all_points3d_matches, all_normals3d_matches, all_invalid_indices, all_normals_scale = self.get_correspondences(warped_vertices, flame_parameters)
        if self.use_normals:
            residual_point2point = ((warped_vertices - all_points3d_matches).abs()).sum(dim=-1)
            residual_point2plane = ((warped_vertices - all_points3d_matches) * all_normals3d_matches).sum(dim=-1).abs() ### why not vertices_normals ???
            #total_geometric_residual  = 0.5 * residual_point2point + 0.5 * residual_point2plane 
            total_geometric_residual = 0.1 * residual_point2point + 0.9 * residual_point2plane
        else:
            total_geometric_residual = ((warped_vertices - all_points3d_matches).abs()).sum(dim=-1)
        total_geometric_residual[all_invalid_indices] = 0
        total_geometric_residual = total_geometric_residual[self.template_mask.unsqueeze(0).repeat(num_frames, 1)]
        total_geometric_residual = total_geometric_residual.sum() / num_frames
        return total_geometric_residual

    def compute_geometric_residual_scan2mesh(self, warped_vertices, flame_parameters):
        # Geometric loss
        num_frames = warped_vertices.shape[0]
        all_points3d_matches, all_normals3d_matches, all_invalid_indices, all_normals_scale, all_vertices_mataches, all_vertices_normals_matches \
            = self.get_correspondences_scan2mesh(warped_vertices, flame_parameters)
        if self.use_normals:
            residual_point2point = ((all_vertices_mataches - all_points3d_matches).abs()).sum(dim=-1)
            residual_point2plane = ((all_vertices_mataches - all_points3d_matches) * all_normals3d_matches).sum(dim=-1).abs()
            total_geometric_residual = 0.1 * residual_point2point + 0.9 * residual_point2plane
        else:
            total_geometric_residual = ((all_vertices_mataches - all_points3d_matches).abs()).sum(dim=-1)
        total_geometric_residual[all_invalid_indices] = 0
        #total_geometric_residual = total_geometric_residual[self.template_mask.unsqueeze(0).repeat(num_frames, 1)]
        total_geometric_residual = total_geometric_residual.sum() / num_frames
        return total_geometric_residual

    def compute_parameter_regularization(self, flame_parameters):
        regularization_shape = flame_parameters['shape'].norm() ** 2
        regularization_expression = flame_parameters['expression'].norm(dim=-1).square().mean()
        regularization_pose = (flame_parameters['jaw_pose'] ** 2).sum(dim=-1).mean()
        if self.use_quanternion:
            quan_iden =  torch.zeros_like(flame_parameters['rotation']).to(self.device)
            quan_iden[:, 0:1] = 1.0
            regularization_rigid = ((flame_parameters['rotation'] - quan_iden) ** 2).sum(dim=-1).mean() + (flame_parameters['translation'] ** 2).sum(dim=-1).mean()
        else:
            regularization_rigid = ((flame_parameters['rotation'] / 2 / math.pi) ** 2).sum(dim=-1).mean() + (flame_parameters['translation'] ** 2).sum(dim=-1).mean()
        regularization_rigid += (flame_parameters['scale'].squeeze() - 1) ** 2
        return {'shape': regularization_shape,
                'expression': regularization_expression,
                'rigid': regularization_rigid,
                'jaw_pose': regularization_pose}
    
    def compute_smoothness_loss(self, flame_parameters):
        num_frames = flame_parameters['expression'].shape[0]
        smoothness_exp = (flame_parameters['expression'][1:num_frames, ...] - flame_parameters['expression'][:num_frames - 1]).norm(dim=-1).square().mean()
        smoothness_rigid = (((flame_parameters['rotation'][1:num_frames, ...] - flame_parameters['rotation'][:num_frames - 1]) / 2 / math.pi) ** 2).sum(
            dim=-1).mean() + ((flame_parameters['translation'][1:num_frames, ...] - flame_parameters['translation'][:num_frames - 1]) ** 2).sum(dim=-1).mean()
        smoothness_pose = ((flame_parameters['jaw_pose'][1:num_frames, ...] - flame_parameters['jaw_pose'][:num_frames - 1]) ** 2).sum(dim=-1).mean()
        return {'expression': smoothness_exp,
                'rigid': smoothness_rigid,
                'jaw_pose': smoothness_pose,
                }

    def forward(self, flame_parameters, step_idx):

        # Reshape to (batch_size, num_vertices, 3)
        #warped_vertices_list, warped_landmark_list = self.warp(flame_parameter) why not transform ??
        warped_vertices_list, warped_landmark_list = self.warp(flame_parameters, transform=True)
        num_frames = len(warped_vertices_list)
        
        print(num_frames)

        # Total loss
        # Landmark loss
        if self.use_landmark:
            total_landmark_residual = self.compute_landmark_residual(warped_landmark_list, flame_parameters)
            total_landmark2d_residual = self.compute_landmark2d_residual(warped_landmark_list, flame_parameters)
        else:
            total_landmark_residual = torch.zeros(1).float().to(self.device)
            total_landmark2d_residual = torch.zeros(1).float().to(self.device)
            
        # Geometric loss  mesh2scan
        total_geometric_residual = self.compute_geometric_residual(warped_vertices_list, flame_parameters)
        
        # Geometric loss  scan2mesh
        #total_geometric_residual = self.compute_geometric_residual_scan2mesh(warped_vertices_list, flame_parameters)
        
        # Parameters regularization
        residual_regularization = self.compute_parameter_regularization(flame_parameters)
        # Smoothness loss
        residual_smoothness = self.compute_smoothness_loss(flame_parameters)

        # Lambda weights
        lambda_landmark2d = 1 if step_idx < 1000 else self.config.lambda_landmark2d #0.0
        lambda_landmark = 1 if step_idx < 1000 else self.config.lambda_landmark #0.0
        lambda_geometric = 1 if step_idx < 100 else self.config.lambda_geometric #0.05 for /, 20 for *

        landmark2d_residual_weighted = lambda_landmark2d * total_landmark2d_residual 
        landmark_residual_weighted = lambda_landmark * total_landmark_residual
        geometric_residual_weighted = total_geometric_residual * lambda_geometric #/ lambda_geometric
        shape_params_regularization_weighted = residual_regularization['shape'] * self.config.reg_shape #/ 10 or * 0.1
        expression_params_regularization_weighted = residual_regularization['expression'] * self.config.reg_expre #/ 10 or * 0.1
        pose_params_regularization_weighted = residual_regularization['jaw_pose']  * self.config.reg_pose #/ 5 or * 0.2
        rigid_transform_params_regularization_weighted = residual_regularization['rigid'] * self.config.reg_rigid #/ 5 or * 0.2

        expression_smoothness = residual_smoothness['expression'] * self.config.smo_expre #/ 10 or * 0.1
        pose_smoothness = residual_smoothness['jaw_pose'] * self.config.smo_pose #/ 5 or * 0.2
        rigid_smoothness = residual_smoothness['rigid'] * self.config.smo_rigid #/ 5 or * 0.1

        total_residual = landmark2d_residual_weighted + landmark_residual_weighted + geometric_residual_weighted + shape_params_regularization_weighted +\
                         expression_params_regularization_weighted + pose_params_regularization_weighted + \
                         rigid_transform_params_regularization_weighted + expression_smoothness + pose_smoothness + \
                         rigid_smoothness
        # Recompute/update the normals
        with torch.no_grad():
            for idx in range(num_frames):
                face_vertices = warped_vertices_list[idx][self.template_mesh_faces, :]
                face_edges1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
                face_edges2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
                normals = torch.cross(face_edges1, face_edges2)

                self.template_vertex_normals_frames[idx].zero_()
                self.template_vertex_normals_frames[idx][self.template_mesh_faces[:, 0], :] += normals
                self.template_vertex_normals_frames[idx][self.template_mesh_faces[:, 1], :] += normals
                self.template_vertex_normals_frames[idx][self.template_mesh_faces[:, 2], :] += normals

                # Normalize the normals
                self.template_vertex_normals_frames[idx] /= torch.norm(self.template_vertex_normals_frames[idx], dim=-1, keepdim=True)

        return {"total_loss": total_residual, 
                "landmark2d_loss": landmark2d_residual_weighted,
                "landmark_loss": landmark_residual_weighted,
                "geometric_loss": geometric_residual_weighted,
                "shape_params_loss": shape_params_regularization_weighted,
                "expression_params_loss": expression_params_regularization_weighted,
                "pose_params_loss": pose_params_regularization_weighted,
                "rigid_transform_loss": rigid_transform_params_regularization_weighted,
                "expression_smoothness": expression_smoothness,
                "pose_smoothness": pose_smoothness,
                "rigid_smoothness": rigid_smoothness,}

    def visualize(self, flame_parameters, show=False, export_dir_base=None, step_idx=123456, visualize=False):
        warped_vertices, warped_landmarks = self.warp(flame_parameters, transform=True)
        unwarped_vertices, unwarped_landmarks = self.warp(flame_parameters, transform=False)
        _, _, all_invalid_indices, _ = self.get_correspondences(warped_vertices, flame_parameters)

        meshes_flame_space = []
        meshes_mvs_space = []
        landmark_flame_space = []
        landmark_mvs_space = []
        
        points_obs = []
        normals_obs = []
        landmark_obs = []
        points_obs_flame_space = []
        normals_obs_flame_space = []
        landmark_obs_flame_space = []
        with torch.no_grad():
            for idx in range(len(flame_parameters['expression'])):
                target_point_cloud = self.all_point3d[idx]
                target_normals = self.all_normals3d[idx]
                scale = flame_parameters['scale']
                translation = flame_parameters['translation'][idx]
                if self.use_quanternion:
                    Rotation = quaternion_to_matrix(flame_parameters['rotation'][idx]).squeeze()
                else:
                    Rotation = euler_angles_to_matrix(flame_parameters['rotation'][idx], self.euler_convention).squeeze()

                if self.from_flame_to_scan:
                    # Transform the target point cloud to Flame space
                    target_point_cloud = 1/scale * (target_point_cloud - translation) @ Rotation
                    target_point_cloud = target_point_cloud.detach().cpu().numpy()
                    target_normals = (target_normals @ Rotation).detach().cpu().numpy()

                    # Transform the landmarks to Flame space
                    landmarks_pointcloud = 1/scale * (self.all_landmarks3d[idx] - translation) @ Rotation
                    landmarks_pointcloud = landmarks_pointcloud.detach().cpu().numpy()
                else:
                    # Transform the target point cloud to Flame space
                    target_point_cloud = scale * target_point_cloud @ Rotation.T + translation
                    target_point_cloud = target_point_cloud.detach().cpu().numpy()
                    target_normals = (target_normals @ Rotation.T).detach().cpu().numpy()

                    # Transform the landmarks to Flame space
                    landmarks_pointcloud = scale * self.all_landmarks3d[idx] @ Rotation.T + translation
                    landmarks_pointcloud = landmarks_pointcloud.detach().cpu().numpy()

                # Predicted landmarks in Flame space
                landmarks_predicted = unwarped_landmarks[idx].detach().cpu().numpy()

                # The meshes without applying any transformation
                mesh_unwarped = self.template_frames[0].copy()
                mesh_unwarped.vertices = unwarped_vertices[idx].detach().cpu().numpy()
                meshes_flame_space.append(mesh_unwarped)
                landmark_flame_space.append(unwarped_landmarks[idx].detach().cpu().numpy())

                # Projecting Flame mesh to MVS space
                mesh_warped = trimesh.Trimesh(warped_vertices[idx].detach().cpu().numpy(), self.template_frames[0].faces, process=False)
                meshes_mvs_space.append(mesh_warped)
                landmark_mvs_space.append(warped_landmarks[idx].detach().cpu().numpy())
                
                if self.mask_foreface:
                    valid_idx = self.mask_foreface_region(flame_parameters, idx)
                    points_valid = self.all_point3d[idx, valid_idx]
                    normals_valid = self.all_normals3d[idx, valid_idx]
                    target_point_cloud_valid =  target_point_cloud[valid_idx.cpu().numpy()]
                    target_normals_valid = target_normals[valid_idx.cpu().numpy()]
                else:
                    points_valid = self.all_point3d[idx, :]
                    normals_valid = self.all_normals3d[idx, :]
                    target_point_cloud_valid =  target_point_cloud
                    target_normals_valid = target_normals
                    
                points_obs.append( points_valid.detach().cpu().numpy() )
                normals_obs.append( normals_valid.detach().cpu().numpy() )
                landmark_obs.append( self.all_landmarks3d[idx].detach().cpu().numpy() )
            
                points_obs_flame_space.append( target_point_cloud_valid)
                normals_obs_flame_space.append( target_normals_valid )
                landmark_obs_flame_space.append( landmarks_pointcloud )
                    

                if visualize:
                    # Visualize comrehensively everything
                    pl = pv.Plotter(shape=(2, 2), off_screen=not show)

                    # Add pointcloud & normals from Colmap
                    pl.subplot(0, 0)
                    pl.add_points(target_point_cloud, render_points_as_spheres=True, point_size=2.5, scalars=((target_normals+1)/2*255).astype(np.uint8), rgb=True)
                    self.set_camera(pl)

                    # Overlay  Flame mesh and points from Colmap
                    pl.subplot(0, 1)
                    pl.add_points(target_point_cloud, render_points_as_spheres=True, point_size=2.5, scalars=((target_normals+1)/2*255).astype(np.uint8), rgb=True)
                    ###pl.add_points(unwarped_vertices[idx].detach().cpu().numpy(), render_points_as_spheres=True, point_size=2.5, scalars=all_invalid_indices[idx].detach().cpu().numpy())
                    pl.add_mesh(mesh_unwarped)
                    self.set_camera(pl)

                    pl.subplot(1, 0)
                    pl.add_points(landmarks_predicted, render_points_as_spheres=True, point_size=2.5, color='red')
                    pl.add_points(landmarks_pointcloud, render_points_as_spheres=True, point_size=2.5, color='blue')
                    for k in range(landmarks_pointcloud.shape[0]):
                        pl.add_mesh(pv.Line(landmarks_predicted[k, :], landmarks_pointcloud[k, :]))
                    self.set_camera(pl)

                    pl.subplot(1, 1)
                    pl.add_mesh(mesh_unwarped)
                    pl.add_points(landmarks_predicted, render_points_as_spheres=True, point_size=2.5, color='red')
                    pl.add_points(landmarks_pointcloud, render_points_as_spheres=True, point_size=2.5, color='blue')
                    for k in range(landmarks_pointcloud.shape[0]):
                        pl.add_mesh(pv.Line(landmarks_predicted[k, :], landmarks_pointcloud[k, :]))
                    self.set_camera(pl)
                    pl.link_views()
                    self.set_camera(pl)

                    if export_dir_base is not None:
                        export_dir = f"{export_dir_base}/landmark_fitting/frame_{idx:05d}/"
                        os.makedirs(export_dir, exist_ok=True)
                        pl.show(screenshot=f"{export_dir}/{step_idx:05d}.png")
            #return meshes_flame_space, meshes_mvs_space
            return meshes_flame_space, landmark_flame_space, \
                   meshes_mvs_space, landmark_mvs_space, \
                       points_obs, normals_obs, landmark_obs, \
                       points_obs_flame_space, normals_obs_flame_space, landmark_obs_flame_space

    def set_camera(self, pl):
        pl.camera_position = (0, 0, 10)
        pl.camera.zoom(1.5)
        pl.camera.roll = 0


