import os
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader

#from datasets.generic import Batch
from model.RAFTSceneFlow import RSF_DGCNN
from tools.loss import sequence_loss, compute_loss
from tools.metric import compute_epe, compute_rmse
from tools.visual import flow_visualize
from tools.save_gnn_results import pred_save
from deepptv.data import FluidflowDataset2D, FluidflowDataset2D_test

import time

def parse_args():
    parser = argparse.ArgumentParser(description='Testing Argument')
    parser.add_argument('--root',
                        help='workspace path',
                        default='',
                        type=str)
    parser.add_argument('--exp_path',
                        help='specified experiment log path',
                        default='test',
                        type=str)
    parser.add_argument('--dataset',
                        help="choose dataset",
                        default='PTVflow2D',
                        type=str)
    parser.add_argument('--max_points',
                        help='maximum number of points sampled from a point cloud',
                        default=8192,
                        type=int)
    parser.add_argument('--corr_levels',
                        help='number of correlation pyramid levels',
                        default=3,
                        type=int)
    parser.add_argument('--base_scales',
                        help='pixelize base scale',
                        default=0.25,
                        type=float)
    parser.add_argument('--truncate_k',
                        help='value of truncate_k in corr block',
                        default=512,
                        type=int)
    parser.add_argument('--iters',
                        help='number of iterations in GRU module',
                        default=12,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus that used for training',
                        default='0,1,2,3',
                        type=str)
    parser.add_argument('--weights',
                        help='checkpoint weights to be loaded',
                        default='./precomputed_ckpts/GNN_dis_predictor/best_checkpoint.params',
                        type=str)
    parser.add_argument('--num_points',
                        type=int, 
                        default=512,
                        help='Point Number [default: 512]')
    parser.add_argument('--test_batch_size',
                        help='number of samples in a test mini-batch',
                        default=1,
                        type=int)
    args = parser.parse_args()

    return args


def testing(args):
    log_dir = os.path.join(args.root, 'experiments', args.exp_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = 'TestAlone_' + args.dataset + datetime.now().strftime("_%m_%d_%H_%M_%S") + '.log'# '_VSJ301'
    logging.basicConfig(
        filename=os.path.join(log_dir, log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO
    )
    warnings.filterwarnings('ignore')
    logging.info(args)
    save_path = 'result/test'

    if args.dataset == 'PTVflow2D':
        folder = 'PTVflow2D'
        dataset_path = os.path.join('data/', folder)
        test_dataset = FluidflowDataset2D_test(npoints=args.num_points, root=dataset_path, partition='test')
    else:
        raise NotImplementedError

    test_dataloader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4,
                                 drop_last=False)

    model = RSF_DGCNN(args).to('cuda')

    weight_path = args.weights
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Load checkpoint from {}'.format(weight_path))
        print('Checkpoint epoch {}'.format(checkpoint['epoch']))
        logging.info('Load checkpoint from {}'.format(weight_path))
        logging.info('Checkpoint epoch {}'.format(checkpoint['epoch']))
    else:
        raise RuntimeError(f"=> No checkpoint found at '{weight_path}")

    model.eval()
    loss_test = []
    epe_test = []
    outlier_test = []
    acc2dRelax_test = []
    acc2dStrict_test = []
    rmse_test = []
    test_progress = tqdm(test_dataloader, ncols=150)
    #tic = time.time()
    for i, batch_data in enumerate(test_progress):
        pc1, pc2, flow = batch_data
        mask = (~torch.isnan(flow)).any(dim=-1)
        batch_data = {"sequence": [pc1, pc2], "ground_truth": [mask, flow]}   
        for key in batch_data.keys():
            batch_data[key] = [d.to('cuda') for d in batch_data[key]]
        with torch.no_grad():
            est_flow = model(batch_data['sequence'], args.iters)

        loss = sequence_loss(est_flow, batch_data)
        epe, acc2d_strict, acc2d_relax, outlier = compute_epe(est_flow[-1], batch_data)
        rmse = compute_rmse(est_flow[-1], batch_data)

        #flow_visualize(est_flow[-1], batch_data)
        #pred_save(est_flow[-1], batch_data, i, args.test_batch_size, save_path)

        loss_test.append(loss.cpu())
        epe_test.append(epe)
        outlier_test.append(outlier)
        acc2dRelax_test.append(acc2d_relax)
        acc2dStrict_test.append(acc2d_strict)
        rmse_test.append(rmse)

        test_progress.set_description(
            'Testing: Loss: {:.5f} EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} RMSE: {:.5f}'.format(
                np.array(loss_test).mean(),
                np.array(epe_test).mean(),
                np.array(outlier_test).mean(),
                np.array(acc2dRelax_test).mean(),
                np.array(acc2dStrict_test).mean(),
                np.array(rmse_test).mean()
            )
        )

    print('Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} RMSE: {:.5f}'.format(
        np.array(epe_test).mean(),
        np.array(outlier_test).mean(),
        np.array(acc2dRelax_test).mean(),
        np.array(acc2dStrict_test).mean(),
        np.array(rmse_test).mean()
    ))
    logging.info(
        'Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} RMSE: {:.5f}'.format(
            np.array(epe_test).mean(),
            np.array(outlier_test).mean(),
            np.array(acc2dRelax_test).mean(),
            np.array(acc2dStrict_test).mean(),
            np.array(rmse_test).mean()
        ))

    #toc = time.time()
    #print('Time:', toc-tic)



if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    #tic = time.time()
    testing(args)
    #toc = time.time()
    #print('Time:', toc-tic)
