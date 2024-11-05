import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CorrBlock2(nn.Module):
    def __init__(self, num_levels=3, base_scale=0.25, resolution=3, truncate_k=128, knn=24):
        super(CorrBlock2, self).__init__()
        self.truncate_k = truncate_k
        self.num_levels = num_levels
        self.resolution = resolution  # local resolution
        self.base_scale = base_scale  # search (base_scale * resolution)^2 square
        self.out_conv = nn.Sequential(
            nn.Conv1d((self.resolution ** 2) * self.num_levels, 64, 1),
            nn.GroupNorm(8, 64),
            nn.PReLU(),
            nn.Conv1d(64, 64, 1)
        )
        self.knn = knn

        self.knn_conv = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.PReLU()
        )

        self.knn_out = nn.Conv1d(64, 64, 1)

    def init_module(self, fmap1, fmap2, xy2, transport):
        # corr = self.calculate_corr(fmap1, fmap2)
        # corr = self.calculate_ncc(fmap1, fmap2)  # cosine similarity
        corr = transport

        b, n_p1, n_p2 = corr.size()
        xy2 = xy2.view(b, 1, n_p2, 2).expand(b, n_p1, n_p2, 2)

        corr_topk = torch.topk(corr.clone(), k=self.truncate_k, dim=2, sorted=True)
        self.truncated_corr = corr_topk.values
        indx = corr_topk.indices.reshape(b, n_p1, self.truncate_k, 1).expand(b, n_p1, self.truncate_k, 2)
        self.ones_matrix = torch.ones_like(self.truncated_corr)
        
        self.truncate_xy2 = torch.gather(xy2, dim=2, index=indx)

    def __call__(self, coords, all_delta_flow, num_iters, scale):
        
        return self.get_adaptive_voxel_feature2(coords, all_delta_flow, num_iters, scale) + self.get_dynamic_knn_feature(coords, all_delta_flow)
    
    def get_adaptive_voxel_feature2(self, coords, all_delta_flow, num_iters, scale):
        b, n_p, _ = coords.size()
        corr_feature = []
        from torch_scatter import scatter_add
        for i in range(self.num_levels):
            r = scale * (2 ** i)
            dis_voxel = torch.round((self.truncate_xy2 - coords.unsqueeze(dim=-2)) / r)  # [B, N, truncate_k, 2]

            ids = len(all_delta_flow)
            #eformable pixel-based correlation retrieval
            if ids >= 2 and ids < (num_iters-2):
                r1_norm = torch.norm(all_delta_flow[-1], dim=-1, keepdim=True)
                r1_max = torch.max(r1_norm, dim=1, keepdim=True)[0]
                cos1 = torch.cosine_similarity(all_delta_flow[-1], all_delta_flow[-2], dim=-1).unsqueeze(dim=-1)
                r1_len = all_delta_flow[-1] / r1_norm * (0.125*r1_norm/r1_max + 0.125*cos1 + 1) * r
                r1 = r1_len.unsqueeze(dim=-2).repeat(1, 1, self.truncate_xy2.shape[2], 1)
                r2_p = all_delta_flow[-2] - cos1 * torch.norm(all_delta_flow[-2], dim=-1, keepdim=True) * all_delta_flow[-1] / r1_norm  # direction vector
                r2_norm = torch.norm(r2_p, dim=-1, keepdim=True)
                r2_max = torch.max(r2_norm, dim=1, keepdim=True)[0]
                r2_len = r2_p / r2_norm * (0.125*r2_norm/r2_max + 0.9) * r
                r2 = r2_len.unsqueeze(dim=-2).repeat(1, 1, self.truncate_xy2.shape[2], 1)
                dis_voxel[:, :, :, 0] = torch.round(torch.sum((self.truncate_xy2 - coords.unsqueeze(dim=-2)) * r1, dim=-1) / torch.norm(r1, dim=-1))
                dis_voxel[:, :, :, 1] = torch.round(torch.sum((self.truncate_xy2 - coords.unsqueeze(dim=-2)) * r2, dim=-1) / torch.norm(r2, dim=-1))

            valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.resolution / 2)).all(dim=-1)
            dis_voxel = dis_voxel - (-1)
            cube_idx = dis_voxel[:, :, :, 0] * self.resolution + dis_voxel[:, :, :, 1]
            cube_idx_scatter = cube_idx.type(torch.int64) * valid_scatter

            valid_scatter = valid_scatter.detach()
            cube_idx_scatter = cube_idx_scatter.detach()

            corr_add = scatter_add(self.truncated_corr * valid_scatter, cube_idx_scatter)
            corr_cnt = torch.clamp(scatter_add(self.ones_matrix * valid_scatter, cube_idx_scatter), 1, n_p)
            corr = corr_add / corr_cnt  # average of the transport plans in each subsquares
            if corr.shape[-1] != self.resolution ** 2:
                repair = torch.zeros([b, n_p, self.resolution ** 2 - corr.shape[-1]], device=coords.device)
                corr = torch.cat([corr, repair], dim=-1)

            corr_feature.append(corr.transpose(1, 2).contiguous())

        return self.out_conv(torch.cat(corr_feature, dim=1))

    def get_dynamic_knn_feature(self, coords, all_delta_flow):
        b, n_p, _ = coords.size()

        dist = self.truncate_xy2 - coords.view(b, n_p, 1, 2)
        dist = torch.sum(dist ** 2, dim=-1)
        # dynamic point-based correlation retrieval
        if len(all_delta_flow) < 10:
            dynamic_k = self.knn - 2 * len(all_delta_flow)
        else:
            dynamic_k = self.knn - 2 * 10
        # fixed point-based correlation retrieval
        # dynamic_k = 16
        # dynamic_k = self.knn - 2 * len(all_delta_flow)
        neighbors = torch.topk(-dist, k=dynamic_k, dim=2).indices

        b, n_p, _ = coords.size()
        knn_corr = torch.gather(self.truncated_corr.view(b * n_p, self.truncate_k), dim=1,
                                index=neighbors.reshape(b * n_p, dynamic_k)).reshape(b, 1, n_p, dynamic_k)

        neighbors = neighbors.view(b, n_p, dynamic_k, 1).expand(b, n_p, dynamic_k, 2)
        knn_xy = torch.gather(self.truncate_xy2, dim=2, index=neighbors).permute(0, 3, 1, 2).contiguous()
        knn_xy = knn_xy - coords.transpose(1, 2).reshape(b, 2, n_p, 1)

        knn_feature = self.knn_conv(torch.cat([knn_corr, knn_xy], dim=1))
        knn_feature = torch.max(knn_feature, dim=3)[0]
        return self.knn_out(knn_feature)

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        return corr

    @staticmethod
    def calculate_ncc(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        n1 = torch.norm(fmap1, dim=1, keepdim=True)
        n2 = torch.norm(fmap2, dim=1, keepdim=True)
        ncc = corr / torch.matmul(n1.transpose(1, 2), n2)
        return ncc
