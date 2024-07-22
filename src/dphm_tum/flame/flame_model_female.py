"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
"""
# Modified from smplx code for FLAME
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from dphm_tum.flame.lbs import lbs, batch_rodrigues, vertices2landmarks
from dphm_tum.utils.flame_utils import rot_mat_to_euler


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(nn.Module):
    """
        Given flame parameters this class generates a differentiable FLAME function
        which outputs the mesh vertices
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        print("Creating the FLAME Decoder")
        with open(config.flame_model_female_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour
        self.use_3D_translation = config.use_3D_translation

        # The faces of the template model
        self.faces = flame_model.f
        self.register_buffer('faces_tensor', to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))

        default_shape = torch.zeros([1, 300 - config.z_shape_flame_female], dtype=self.dtype, requires_grad=False,)
        self.register_parameter("shape_betas", nn.Parameter(default_shape, requires_grad=False))

        default_exp = torch.zeros([1, 100 - config.z_expression_flame_female], dtype=self.dtype, requires_grad=False,)
        self.register_parameter("expression_betas", nn.Parameter(default_exp, requires_grad=False))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))

        default_transl = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter("transl", nn.Parameter(default_transl, requires_grad=False))

        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        self.register_buffer('shapedirs', shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))

        # The joint regressor
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

        # Static and Dynamic Landmark embeddings for FLAME
        with open(config.flame_static_landmark_embedding_path, "rb") as f:
            static_embeddings = Struct(**pickle.load(f, encoding="latin1"))

        lmk_faces_idx = static_embeddings.lmk_face_idx.astype(np.int64)
        self.register_buffer("lmk_faces_idx", torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer("lmk_bary_coords", torch.tensor(lmk_bary_coords, dtype=self.dtype))

        if self.use_face_contour:
            conture_embeddings = np.load(config.flame_dynamic_landmark_embedding_path, allow_pickle=True, encoding="latin1",)
            conture_embeddings = conture_embeddings[()]
            dynamic_lmk_faces_idx = np.array(conture_embeddings["lmk_face_idx"]).astype(np.int64)
            dynamic_lmk_faces_idx = torch.tensor(dynamic_lmk_faces_idx, dtype=torch.long)
            self.register_buffer("dynamic_lmk_faces_idx", dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = conture_embeddings["lmk_b_coords"]
            dynamic_lmk_bary_coords = np.array(dynamic_lmk_bary_coords)
            dynamic_lmk_bary_coords = torch.tensor(dynamic_lmk_bary_coords, dtype=self.dtype)
            self.register_buffer("dynamic_lmk_bary_coords", dynamic_lmk_bary_coords)

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32,):
        """
        Selects the face contour depending on the reletive position of the head
        Input:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        Source: Modified for batches from https://github.com/vchoutas/smplx
        """
        batch_size = vertices.shape[0]
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        rel_rot_mat = (torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1))
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose_params=None, eye_pose_params=None, full_pose=None, transl=None):
        """
            Input:
                shape_params: N X number of shape parameters (100)
                expression_params: N X number of expression parameters (50)
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """

        batch_size = shape_params.shape[0]
        betas = torch.cat([shape_params, self.shape_betas, expression_params, self.expression_betas], dim=1)

        transl = transl if transl is not None else self.transl
        transl = transl.repeat(batch_size, 1)
        neck_pose_params = neck_pose_params if neck_pose_params is not None else self.neck_pose
        eye_pose_params = eye_pose_params if eye_pose_params is not None else self.eye_pose
        full_pose = torch.cat([pose_params[:, :3], neck_pose_params.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params.expand(batch_size, -1)], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # import ipdb; ipdb.set_trace()
        vertices, _ = lbs(betas, full_pose, template_vertices, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        if self.use_face_contour:
            (dyn_lmk_faces_idx,dyn_lmk_bary_coords,) = self._find_dynamic_lmk_idx_and_bcoords(vertices, full_pose, self.dynamic_lmk_faces_idx, self.dynamic_lmk_bary_coords, self.neck_kin_chain, dtype=self.dtype,)

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        if self.use_3D_translation:
            landmarks += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
        return vertices, landmarks


