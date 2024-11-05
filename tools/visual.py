import numpy as np
import torch
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf, RBFInterpolator

def flow_visualize(est_flow, batch):
    """
    Draws a comparison plot between actual and predicted vector groups.

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

    fig, axs = plt.subplots(m, 1, figsize=(m * 5, m * 5))
    if m == 1:
        axs = [axs]
    for i in range(m):
        X1, Y1 = pc1[i, :, 0], pc1[i, :, 1]
        X2, Y2 = pc2[i, :, 0], pc2[i, :, 1]
        U_gt, V_gt = sf_gt[i, :, 0], sf_gt[i, :, 1]
        U_pred, V_pred = sf_pred[i, :, 0], sf_pred[i, :, 1]
        #axs[i].scatter(X1, Y1, color='blue', s=2)
        #axs[i].scatter(X2, Y2, color='green', s=2)
        axs[i].quiver(X1, Y1, U_gt, V_gt, angles='xy', scale_units='xy', scale=1, color='red', label='Actual', minshaft=3)
        axs[i].quiver(X1, Y1, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color='blue', label='Predicted', minshaft=3)
        # Set x and y axis limits
        min_x, max_x = np.min(pc1[i, :, 0]), np.max(pc1[i, :, 0])
        min_y, max_y = np.min(pc1[i, :, 1]), np.max(pc1[i, :, 1])
        axs[i].set_xlim(min_x - 1, max_x + 1)
        axs[i].set_ylim(min_y - 1, max_y + 1)
        axs[i].set_aspect(1)
        axs[i].set_title('')
    plt.gca().invert_yaxis()
    plt.show()

def flow_pic_test(est_flow, batch, image, x_positions1, y_positions1):
    """
    Draws a comparison plot between actual and predicted vector groups.

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
    sf_pred = est_flow.cpu().numpy()#[mask > 0]
    pc1 = batch["sequence"][0].cpu().numpy()#
    pc2 = batch["sequence"][1].cpu().numpy()
    #pc1_gt = pc1[mask > 0]
    m=1


    fig, axs = plt.subplots(m, 1, figsize=(m * 5, m * 5))
    if m == 1:  #
        axs = [axs]
    for i in range(m):
        X1, Y1 = pc1[i, :, 0], pc1[i, :, 1]  
        X2, Y2 = pc2[i, :, 0], pc2[i, :, 1]

        U_pred, V_pred = sf_pred[i, :, 0], sf_pred[i, :, 1]  # pred

        axs[i].quiver(X1, Y1, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color='red', minshaft=3)
        min_x, max_x = np.min(pc1[i, :, 0]), np.max(pc1[i, :, 0])
        min_y, max_y = np.min(pc1[i, :, 1]), np.max(pc1[i, :, 1])
        axs[i].set_xlim(min_x - 1, max_x + 1)
        axs[i].set_ylim(min_y - 1, max_y + 1)

        handles, labels = axs[i].get_legend_handles_labels()

    axs[0].imshow(image, cmap='gray')  
    axs[0].scatter(x_positions1, y_positions1, facecolors='none', edgecolors='green', s=20)
                
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()