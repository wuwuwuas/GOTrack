import torch
from scipy.interpolate import griddata, Rbf, RBFInterpolator
import numpy as np
from knn_cuda import KNN


def similarity_verify(pclouds, track, neighbors, thres, neigh_p0, neigh_p1):

    track_mask = (track != -1) 
    idx = track  # B,N
    neigh_k = neighbors

    neigh_p0 = neigh_p0[..., 1:neigh_k + 1]  
    neigh_p1 = neigh_p1[..., 1:neigh_k + 1]

    idx_full = idx * track_mask
    replace_value = torch.full_like(neigh_p0, -2)

    x1n = torch.gather(neigh_p1, 1, idx_full.unsqueeze(-1).expand(-1, -1, neigh_k))#b,np,nk
    x1n = torch.where(track_mask.unsqueeze(-1).expand_as(x1n), x1n, replace_value)

    num = torch.zeros(1, x1n.shape[1]).to(x1n.device)  # B,NT

    for i in range(neigh_k):
        x0n = torch.gather(idx, 1, neigh_p0[:, :, i]) 
        num = torch.any(torch.eq(x1n, x0n.unsqueeze(-1)), dim=-1).long() + num  

    sim_mask = num >= thres  # B,N
    mask = track_mask * sim_mask
    valid_idx = idx * (mask != 0) + (-1) * (mask == 0)
    return valid_idx


def removeoutlier(pclouds, track, neighbors, thres):
    pcloud0 = pclouds[0]
    b, n, _ = pcloud0.shape
    neigh_k = neighbors
    track_mask = (track != 0)  # 

    idx = track - 1
    idx_full = idx * track_mask
    idx_full_expand = idx_full.unsqueeze(-1).expand(-1, -1, 2)
    pcloud1 = torch.gather(pclouds[1], 1, idx_full_expand)
    flow = pcloud1 - pcloud0

    knn = KNN(k=neigh_k, transpose_mode=True)

    for i in range(b):
        pc_0 = pcloud0[i, :, :]

        tracki = track[i, :]
        track_maski = track_mask[i, :]
        flowi = flow[i, :, :]

        idxi = tracki[track_maski]
        pc_0 = pc_0[track_maski].unsqueeze(0)
        flowi = flowi[track_maski]


        _, neigh_p = knn(pc_0, pc_0)

        flow_nu = flowi[neigh_p.squeeze(0), 0]
        flow_nv = flowi[neigh_p.squeeze(0), 1]

        flow_mu, _ = torch.median(flow_nu, dim=-1)
        flow_mv, _ = torch.median(flow_nv, dim=-1)

        e = 0.075
        r1 = torch.abs(flow_nu - flow_mu.unsqueeze(-1))
        r1, _ = torch.median(r1, dim=-1)
        r1 = r1 + e
        r2 = torch.abs(flow_nv - flow_mv.unsqueeze(-1))
        r2, _ = torch.median(r2, dim=-1)
        r2 = r2 + e

        rn1 = torch.abs(flowi[:, 0] - flow_mu)
        rn1 = rn1 / r1
        rn2 = torch.abs(flowi[:, 1] - flow_mv)
        rn2 = rn2 / r2

        ridx1 = rn1 > thres
        ridx2 = rn2 > thres

        IDX = ridx1 | ridx2 #| ridx3

        idxi[IDX] = 0

        tracki[track_maski] = idxi
        track[i, :] = tracki

    return track, flow

def griddata_flow(pclouds, flow, trackmask):

    b, n = trackmask.shape
    flow_zero = trackmask != 1
    flow_value = trackmask

    for i in range(b):
        pc0_flow_zeros = pclouds[i, :, :][flow_zero[i, :]].detach().cpu().numpy() 
        pc0_inter_coordinates = pclouds[i, :, :][flow_value[i, :]].detach().cpu().numpy()  
        flow0 = flow[i, :, :][flow_value[i, :]].detach().cpu().numpy()
        inter_flow = np.zeros_like(pc0_flow_zeros)

        inter_flow = RBFInterpolator(pc0_inter_coordinates, flow0, neighbors=32)(pc0_flow_zeros).astype(np.float32)

        flow[i, :, :][flow_zero[i, :]] = torch.tensor(inter_flow).cuda()
    return flow


def get_recon_flow(transport_cross, pclouds):  

    row_second_max_values, row_second_max_indices = torch.topk(transport_cross, k=2, dim=-1)  # B,N,2
    column_second_max_values, column_second_max_indices = torch.topk(transport_cross, k=2, dim=-2)  # B,2,N

    row_max_indices_expanded = row_second_max_indices[:, :, 0].unsqueeze(2).expand(-1, -1, 2)
    pc1 = torch.gather(pclouds[1], 1, row_max_indices_expanded)
    flow = pc1 - pclouds[0]  # B*N*2

    row = torch.arange(pclouds[0].shape[1]).unsqueeze(0).cuda()  # 1*N
    condition_indices = row == torch.gather(column_second_max_indices[:, 0, :], 1, row_second_max_indices[:, :, 0])


    flow = torch.where(condition_indices.unsqueeze(-1), flow, torch.zeros_like(flow))
    return flow, column_second_max_indices, row_second_max_indices, condition_indices


def get_recon_flow_knn(distance, pclouds):  

    row_second_min_values, row_second_min_indices = torch.topk(distance, k=2, dim=-1, largest=False)  # B,N,2
    column_second_min_values, column_second_min_indices = torch.topk(distance, k=2, dim=-2, largest=False)  # B,2,N

    row_min_indices_expanded = row_second_min_indices[:, :, 0].unsqueeze(2).expand(-1, -1, 2)
    pc1 = torch.gather(pclouds[1], 1, row_min_indices_expanded)
    flow = pc1 - pclouds[0]  # B*N*2

    row = torch.arange(pclouds[0].shape[1]).unsqueeze(0).cuda()  # 1*N
    condition_indices = row == torch.gather(column_second_min_indices[:, 0, :], 1, row_second_min_indices[:, :, 0])

    flow = torch.where(condition_indices.unsqueeze(-1), flow, torch.zeros_like(flow))
    return flow, column_second_min_indices, row_second_min_indices, condition_indices

