import numpy as np
import torch
import os

def pred_save(est_flow, batch, batchnum, batch_size, folderpath):
    """
    save results: pos1, pos2, pred, ground truth
    as npz

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    None

    """

    # Extract occlusion mask
    #mask = batch["ground_truth"][0].cpu().numpy()#[..., 0]

    # Flow
    sf_gt = batch["ground_truth"][1].cpu().numpy()
    sf_pred = est_flow.cpu().numpy()
    pc1 = batch["sequence"][0].cpu().numpy()
    pc2 = batch["sequence"][1].cpu().numpy()
    m, n, _ = sf_pred.shape

    for i in range(m):

        X1, Y1 = pc1[i, :, 0], pc1[i, :, 1]
        X2, Y2 = pc2[i, :, 0], pc2[i, :, 1]
        U_gt, V_gt = sf_gt[i, :, 0], sf_gt[i, :, 1]
        U_pred, V_pred = sf_pred[i, :, 0], sf_pred[i, :, 1]

        count = batchnum * batch_size + i

        filename = f"TESTGnnResult_{count:04d}.npz"
        path = os.path.join(folderpath, filename)
        np.savez(path, x1=X1, y1=Y1, x2=X2, y2=Y2, U_gt=U_gt, V_gt=V_gt, U_pred=U_pred, V_pred=V_pred)
