import pytorch3d
import torch
import sys
import os
import numpy as np
from editor.gaussiansplatting.scene import GaussianModel
from editor.hundrededitorgui.utils.lap_deform import LapDeform
from editor.hundrededitorgui.utils.time_utils import farthest_point_sample
from editor.hundrededitorgui.utils.graphics_utils import BasicPointCloud

svd = torch.svd

class DeformKeypoints:
    def __init__(self) -> None:
        self.keypoints3d_list = []  # list of keypoints group
        self.keypoints_idx_list = [] # keypoints index
        self.keypoints3d_delta_list = []
        self.selective_keypoints_idx_list = []  # keypoints index
        self.idx2group = {}

        self.selective_rotation_keypoints_idx_list = []
        # self.rotation_idx2group = {}

    def get_kpt_idx(self,):
        return self.keypoints_idx_list
    
    def get_kpt(self,):
        return self.keypoints3d_list
    
    def get_kpt_delta(self,):
        return self.keypoints3d_delta_list
    
    def get_deformed_kpt_np(self, rate=1.):
        return np.array(self.keypoints3d_list) + np.array(self.keypoints3d_delta_list) * rate

    def add_kpts(self, keypoints_coord, keypoints_idx, expand=False):
        # keypoints3d: [N, 3], keypoints_idx: [N,], torch.tensor
        # self.selective_keypoints_idx_list.clear()
        selective_keypoints_idx_list = [] if not expand else self.selective_keypoints_idx_list
        for idx in range(len(keypoints_idx)):
            if not self.contain_kpt(keypoints_idx[idx].item()):
                selective_keypoints_idx_list.append(len(self.keypoints_idx_list))
                self.keypoints_idx_list.append(keypoints_idx[idx].item())
                self.keypoints3d_list.append(keypoints_coord[idx].cpu().numpy())            
                self.keypoints3d_delta_list.append(np.zeros_like(self.keypoints3d_list[-1]))

        for kpt_idx in keypoints_idx:
            self.idx2group[kpt_idx.item()] = selective_keypoints_idx_list

        self.selective_keypoints_idx_list = selective_keypoints_idx_list

    def contain_kpt(self, idx):
        # idx: int
        if idx in self.keypoints_idx_list:
            return True
        else:
            return False
        
    def select_kpt(self, idx):
        # idx: int
        # output: idx list of this group
        if idx in self.keypoints_idx_list:
            self.selective_keypoints_idx_list = self.idx2group[idx]

    def select_rotation_kpt(self, idx):
        if idx in self.keypoints_idx_list:
            self.selective_rotation_keypoints_idx_list = self.idx2group[idx]

    def get_rotation_center(self,):
        selected_rotation_points = self.get_deformed_kpt_np()[self.selective_rotation_keypoints_idx_list]
        return selected_rotation_points.mean(axis=0)
    
    def get_selective_center(self,):
        selected_points = self.get_deformed_kpt_np()[self.selective_keypoints_idx_list]
        return selected_points.mean(axis=0)

    def delete_kpt(self, idx):
        for kidx in self.selective_keypoints_idx_list:
            list_idx = self.idx2group.pop(kidx)
        self.keypoints3d_delta_list.pop(list_idx)
        self.keypoints3d_list.pop(list_idx)
        self.keypoints_idx_list.pop(list_idx)

    def delete_batch_ktps(self, batch_idx):
        pass

    def update_delta(self, delta):
        # delta: [3,], np.array
        for idx in self.selective_keypoints_idx_list:
            self.keypoints3d_delta_list[idx] += delta

    def set_delta(self, delta):
        # delta: [N, 3], np.array
        for id, idx in enumerate(self.selective_keypoints_idx_list):
            self.keypoints3d_delta_list[idx] = delta[id]


    def set_rotation_delta(self, rot_mat):
        kpts3d = self.get_deformed_kpt_np()[self.selective_keypoints_idx_list]
        kpts3d_mean = kpts3d.mean(axis=0)
        kpts3d = (kpts3d - kpts3d_mean) @ rot_mat.T + kpts3d_mean
        delta = kpts3d - np.array(self.keypoints3d_list)[self.selective_keypoints_idx_list]
        for id, idx in enumerate(self.selective_keypoints_idx_list):
            self.keypoints3d_delta_list[idx] = delta[id]

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

class NodeDriver:
    def __init__(self):
        pass
    
    def p2dR(self, p, p0, K=8, as_quat=True):
        p = p.detach()
        nn_weight, nn_dist, nn_idx = self.cal_nn_weight(p0, p0, K=K, XisNode=True, cache_target='node')
        edges = torch.gather(p0[:, None].expand([p0.shape[0], K, p0.shape[-1]]), dim=0, index=nn_idx[..., None].expand([p0.shape[0], K, p0.shape[-1]])) - p0[:, None]
        t0_deform = None
        edges_t = torch.gather(p[:, None].expand([p.shape[0], K, p.shape[-1]]), dim=0, index=nn_idx[..., None].expand([p.shape[0], K, p.shape[-1]])) - p[:, None]
        edges, edges_t = edges / (edges.norm(dim=-1, keepdim=True) + 1e-5), edges_t / (edges_t.norm(dim=-1, keepdim=True) + 1e-5)
        W = torch.zeros([edges.shape[0], K, K], dtype=torch.float32, device=edges.device)
        W[:, range(K), range(K)] = nn_weight
        S = torch.einsum('nka,nkg,ngb->nab', edges, W, edges_t)
        U, _, V = svd(S.float())
        dR = torch.matmul(V, U.permute(0, 2, 1))
        if as_quat:
            dR = matrix_to_quaternion(dR)
        return dR, t0_deform
    
    def geodesic_distance_floyd(self, cur_node, K=3):
        node_num = cur_node.shape[0]
        nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(cur_node[None], cur_node[None], None, None, K=K+1)
        nn_dist, nn_idx = nn_dist[0]**.5, nn_idx[0]
        dist_mat = torch.inf * torch.ones([node_num, node_num], dtype=torch.float32, device=cur_node.device)
        dist_mat.scatter_(dim=1, index=nn_idx, src=nn_dist)
        dist_mat = torch.minimum(dist_mat, dist_mat.T)
        for i in range(nn_dist.shape[0]):
            dist_mat = torch.minimum((dist_mat[:, i, None] + dist_mat[None, i, :]), dist_mat)
        return dist_mat
        
    def cal_nn_weight(self, x:torch.Tensor, nodes, K=None, method='floyd', XisNode=False, node_radius=1., cache_target=None, force=False):
        if force or cache_target is None or not hasattr(self, f'cached_{cache_target}') or not getattr(self, f'cached_{cache_target}'):
            if method == 'floyd':
                print(f'Use floyd distance, which is better for topological change!: {cache_target}')
                node_dist_mat = self.geodesic_distance_floyd(cur_node=nodes, K=2)
                floyd_nn_dist, floyd_nn_idx = node_dist_mat.sort(dim=1)
                offset = 1 if XisNode else 0
                node_nn_dist = floyd_nn_dist[:, offset:K+offset]
                node_nn_idxs = floyd_nn_idx[:, offset:K+offset]
                nn1_dist, nn1_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=1)  # N, 1
                nn1_dist, nn1_idxs = nn1_dist[0, :, 0], nn1_idxs[0, :, 0]  # N
                nn_idxs = node_nn_idxs[nn1_idxs]  # N, K
                nn_dist = node_nn_dist[nn1_idxs] + nn1_dist[:, None]  # N, K
            else:
                print(f'Use euclidean distance to calculate the nearest neighbor weight!')
                K = self.K if K is None else K
                K = K + 1 if XisNode else K  # +1 for the node itself
                # Weights of control nodes
                nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)  # N, K
                nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K'
                if XisNode:
                    nn_dist, nn_idxs = nn_dist[:, 1:], nn_idxs[:, 1:]  # N, K
            nn_weight = torch.exp(- nn_dist / (2 * node_radius ** 2))  # N, K
            nn_weight = nn_weight + 1e-7
            nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
            if cache_target is not None:
                setattr(self, f'cached_{cache_target}', True)
                setattr(self, f'{cache_target}_nn_weights_dist_idxs', [nn_weight, nn_dist, nn_idxs])
        else:
            nn_weight, nn_dist, nn_idxs = getattr(self, f'{cache_target}_nn_weights_dist_idxs')
        return nn_weight, nn_dist, nn_idxs

    @torch.no_grad()
    def __call__(self, x, nodes, node_trans_bias, node_radius=1.):
        
        x = x.detach()
        rot_bias = torch.tensor([1., 0, 0, 0]).float().to(x.device)
        # Animation
        return_dict = {'d_xyz': torch.zeros_like(x), 'd_rotation': 0., 'd_scaling': 0.}
        # Initial nodes and gs
        init_node = nodes
        init_gs = x
        init_nn_weight, _, init_nn_idx = self.cal_nn_weight(x=init_gs, nodes=init_node, K=4, node_radius=node_radius, XisNode=False, cache_target='gs')
        # New nodes and gs
        nodes_t = init_node + node_trans_bias
        node_rot_bias, _ = self.p2dR(p=nodes_t, p0=init_node, K=4, as_quat=True)
        d_nn_node_rot_R = quaternion_to_matrix(node_rot_bias)[init_nn_idx]
        # Aligh the relative distance considering the rotation
        gs_t = nodes_t[init_nn_idx] + torch.einsum('gkab,gkb->gka', d_nn_node_rot_R, (init_gs[:, None] - init_node[init_nn_idx]))
        gs_t_avg = (gs_t * init_nn_weight[..., None]).sum(dim=1)
        translate = gs_t_avg - x
        return_dict['d_xyz'] = translate
        return_dict['d_rotation_bias'] = ((node_rot_bias[init_nn_idx] * init_nn_weight[..., None]).sum(dim=1) - rot_bias) + rot_bias
        return_dict['d_opacity'] = None
        return_dict['d_color'] = None
        return return_dict

@torch.no_grad()
def animation_initialize(ply_file):
    gaussian = GaussianModel(
        sh_degree=0,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2,
    )
    # load
    gaussian.load_ply(ply_file)
    gaussian.max_radii2D = torch.zeros(
        (gaussian.get_xyz.shape[0]), device="cuda"
    )
    mask = (gaussian.get_opacity > 0.90)[:, 0]
    pcl = gaussian.get_xyz[mask]

    pts_idx = farthest_point_sample(pcl[None], 512)[0]
    pcl = pcl[pts_idx]
    scale = torch.norm(pcl.max(0).values - pcl.min(0).values)
    node_radius = scale /20
    print(f'Static scene node radius: {node_radius}')
    
    control_nodes = pcl
    animate_tool = LapDeform(init_pcl=pcl, K=4, trajectory=None, node_radius=node_radius)
    keypoint_idxs = []
    keypoint_3ds = []
    keypoint_labels = []
    keypoint_3ds_delta = []
    keypoint_idxs_to_drag = []
    deform_keypoints = DeformKeypoints()
    animation_trans_bias = None
    animation_rot_bias = None
    buffer_overlay = None

    control_nodes_gaussians = GaussianModel(        
            sh_degree=0,
            anchor_weight_init_g0=0,
            anchor_weight_init=0,
            anchor_weight_multiplier=0,
        )
    colors = control_nodes.clone()
    colors = (colors - colors.min(0).values) / (colors.max(0).values - colors.min(0).values)
    pcd = BasicPointCloud(points=pcl.detach().cpu().numpy(), colors=colors.detach().cpu().numpy(), normals=None)
    control_nodes_gaussians.create_from_pcd(pcd=pcd, spatial_lr_scale=5)

    animator = NodeDriver()
    print('Initialize Animation Model with %d control nodes' % len(pcl))

    return {
        'control_nodes': control_nodes,
        'animate_tool': animate_tool,
        'keypoint_idxs': keypoint_idxs,
        'keypoint_3ds': keypoint_3ds,
        'keypoint_labels': keypoint_labels,
        'keypoint_3ds_delta': keypoint_3ds_delta,
        'keypoint_idxs_to_drag': keypoint_idxs_to_drag,
        'deform_keypoints': deform_keypoints,
        'animation_trans_bias': animation_trans_bias,
        'animation_rot_bias': animation_rot_bias,
        'buffer_overlay': buffer_overlay,
        'control_nodes_gaussians': control_nodes_gaussians,
        'animator': animator,
        'gaussians_xyz': gaussian.get_xyz,
    }

def animation_reset():
    reset_dict = {
        'keypoint_idxs': [],
        'keypoint_3ds': [],
        'keypoint_labels': [],
        'keypoint_3ds_delta': [],
        'keypoint_idxs_to_drag': [],
        'deform_keypoints': DeformKeypoints(),
        'animation_trans_bias': None,
        'animation_rot_bias': None,
        'buffer_overlay': None,
        'motion_animation_d_values': None,
        'animator': NodeDriver()
    }
    print('Reset Animation Model ...')
    return reset_dict