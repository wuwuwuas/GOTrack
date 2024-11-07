import numpy as np
import torch
import glob
from PIL import Image
from model.detector import UNet
import os
from model.locator import FCNN
import matplotlib.pyplot as plt
import argparse
import logging
from model.ParticleMatch import Tracking
from tools.visual import flow_pic_test
from scipy.io import savemat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='index of gpu used')

    parser.add_argument('--detector_path', type=str, default='./precomputed_ckpts/CNN_detector/ckpt.tar',
                        help='path of already trained checkpoint for CNN detector (pixel level)')
    parser.add_argument('--locator_path', type=str, default='./precomputed_ckpts/CNN_locator/ckpt.tar',
                        help='path of already trained checkpoint for CNN locator (sub-pixel level)')
    parser.add_argument('--weights_GNN', help='checkpoint weights for GNN to be loaded',
                        default='./precomputed_ckpts/GNN_dis_predictor/best_checkpoint.params', type=str)

    parser.add_argument('--delta_t', type=int, default=1,
                        help='calculation interval for particle tracking')
    parser.add_argument('--illumination', type=float, default=0.9,
                        help='brightness adjustment')
    parser.add_argument('--scale_image', type=float, default=1,
                        help='image resizing to fit the particle size')

    parser.add_argument('--output_dir_ckpt', type=str, default='./result/',
                        help='output directory for results')
    parser.add_argument('--data_path', type=str, default='./data/vsj301/',
                        help='dataset for test')
    parser.add_argument('--img_type', type=str, default='bmp',
                        help='image format')
    parser.add_argument('--name', type=str, default='vsj301',
                        help='name of experiment')
    parser.add_argument('--result_plot', type=int, default=1,
                        choices=[0, 1],
                        help='Whether to plot the results. 0: No; 1: Yes')

    parser.add_argument('--max_points', help='maximum number of points sampled from a point cloud',
                        default=10000,type=int)
    parser.add_argument('--corr_levels', help='number of correlation pyramid levels', default=3, type=int)
    parser.add_argument('--base_scales', help='voxelize base scale', default=0.25,  type=float)
    parser.add_argument('--truncate_k', help='value of truncate_k in corr block', default=512, type=int)
    parser.add_argument('--iters', help='number of iterations in GRU module', default=12, type=int)
    parser.add_argument('--nb_iter', type=int, default=30,
                        help='Number of unrolled iterations in the Sinkhorn algorithm')

    parser.add_argument('--tracking_mode', type=str, default='GOTrack+',
                        choices=['GOTrack+', 'GOTrack'],
                        help='output directory for results')
    parser.add_argument('--candidates', type=int, default=12,
                        help='Number of candidate particles')
    
    parser.add_argument('--neighbor_similarity', type=int, default=7,
                        help='Number of neighbor particles for similarity checking')
    parser.add_argument('--threshold_similarity', type=int, default=4,
                        help='Threshold for similarity checking')  
    
    parser.add_argument('--neighbor_outlier', type=int, default=8,
                        help='Number of neighbor particles for outlier removal')
    parser.add_argument('--threshold_outlier', type=int, default=2,
                        help='Threshold for outlier removal')  
    

    args = parser.parse_args()
    print('args parsed')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    evaluate(args)

def data_read(images_pairs, scale, illumination):
    images = []
    for img_path in images_pairs:
        img = Image.open(img_path).convert('L')

        width, height = img.size
        img_resized = img.resize((int(width // scale), int(height // scale)))
        img = np.array(img_resized, dtype=np.float32)

        # img  = cv2.GaussianBlur(img, (7, 7), 2)
        img = img / img.max() * illumination
        img[img < 0 / 255.0] = 0
        height, width = img.shape

        new_height = (height // 8) * 8
        new_width = (width // 8) * 8

        new_height = new_height - new_height % 2
        new_width = new_width - new_width % 2

        cropped_image = img[:new_height, :new_width]
        images.append(cropped_image)
    return images

def img_split(img, target_img):
    img = img.cpu().numpy()
    target_img = target_img.cpu().numpy()
    img_padded = np.pad(img, pad_width=3, mode='constant', constant_values=0)
    particle_in_img = np.where(target_img == 1.0)
    y_positions = particle_in_img[0]
    x_positions = particle_in_img[1]

    patches = extract_patches_no_loop(img_padded, y_positions + 3, x_positions + 3)

    return patches, y_positions, x_positions

def extract_patches_no_loop(img, x_positions, y_positions, patch_size=7):
    H, W = img.shape
    N = len(x_positions)
    half_size = patch_size // 2

    x_idx = x_positions[:, None, None] + np.arange(-half_size, half_size + 1)[None, :, None]
    y_idx = y_positions[:, None, None] + np.arange(-half_size, half_size + 1)[None, None, :]

    x_idx = np.clip(x_idx, 0, H - 1)
    y_idx = np.clip(y_idx, 0, W - 1)

    patches = img[x_idx, y_idx]

    return patches

def evaluate(args):
    device = 'cuda:' + str(args.gpu)

    model = UNet(in_channels=1, out_channels=1).to(device)
    locator = FCNN().to(device)
    image_path = args.data_path + '*.' + args.img_type

    images_pairs = glob.glob(image_path)

    images_pairs = sorted(images_pairs)
    
    print('\n'.join(images_pairs))
    images = data_read(images_pairs, args.scale_image, args.illumination)
    images = np.array(images)
    images = torch.from_numpy(images).to(torch.float32)

    test_images = images.unsqueeze(1)

    print('total number of test samples:', test_images.shape[0])

    save_path = args.output_dir_ckpt + args.name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = torch.load(args.detector_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint_locator = torch.load(args.locator_path, map_location=device)
    locator.load_state_dict(checkpoint_locator['model_state_dict'])
    
    tracker = Tracking(args).to(device)
    weight_path = args.weights_GNN
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        tracker.load_state_dict(checkpoint['state_dict'])
        print('Load checkpoint from {}'.format(weight_path))
        print('Checkpoint epoch {}'.format(checkpoint['epoch']))
        logging.info('Load checkpoint from {}'.format(weight_path))
        logging.info('Checkpoint epoch {}'.format(checkpoint['epoch']))
    else:
        raise RuntimeError(f"=> No checkpoint found at '{weight_path}")
    # Validation
    with torch.no_grad():
        model.eval()
        locator.eval()
        with torch.no_grad():
            for i in range(0, test_images.shape[0]-args.delta_t, args.delta_t):
                # detection
                print('Sample:   ', i)
                print('------------ Particle Detection--------------')
                # first frame
                image = test_images[i:i+1,...]
                    
                image = image.to(device)
                pred_images = model(image)

                pred_images = torch.where(pred_images > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
                patches, y_positions, x_positions = img_split(image.squeeze(0).squeeze(0), pred_images.squeeze(0).squeeze(0))

                patches = torch.from_numpy(patches).to(device)
                y_positions = torch.from_numpy(y_positions).to(device)
                x_positions = torch.from_numpy(x_positions).to(device)

                pred_pos = locator(patches.unsqueeze(1))
                pred_y = pred_pos[:, 0:1]
                pred_x = pred_pos[:, 1:2]
                pos_pred = torch.cat((pred_y, pred_x),dim=1) + torch.cat((y_positions.unsqueeze(-1), x_positions.unsqueeze(-1)),dim=1)

                image_np1 = image.cpu().squeeze().numpy()
                y_positions1 = pos_pred[:, 0].cpu().numpy()
                x_positions1 = pos_pred[:, 1].cpu().numpy()

                pos1 = np.stack((x_positions1, y_positions1), axis=1)

                print('Number of identified particles in the first frame:  ', x_positions.shape[0])
                
                # second frame
                image = test_images[i+args.delta_t:i+args.delta_t+1,...]

                image = image.to(device)
                pred_images = model(image)

                pred_images = torch.where(pred_images > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
                patches, y_positions, x_positions = img_split(image.squeeze(0).squeeze(0), pred_images.squeeze(0).squeeze(0))

                patches = torch.from_numpy(patches).to(device)
                y_positions = torch.from_numpy(y_positions).to(device)
                x_positions = torch.from_numpy(x_positions).to(device)

                pred_pos = locator(patches.unsqueeze(1))
                pred_y = pred_pos[:, 0:1]
                pred_x = pred_pos[:, 1:2]
                pos_pred = torch.cat((pred_y, pred_x),dim=1) + torch.cat((y_positions.unsqueeze(-1), x_positions.unsqueeze(-1)),dim=1)

                image_np2 = image.cpu().squeeze().numpy()
                y_positions2 = pos_pred[:, 0].cpu().numpy()
                x_positions2 = pos_pred[:, 1].cpu().numpy()

                pos2 = np.stack((x_positions2, y_positions2), axis=1)

                print('Number of identified particles in the second frame:  ', x_positions.shape[0])

                if args.result_plot == 1 :
                    fig, axs = plt.subplots(1, 2, figsize=(16, 8)) 
                    axs[0].imshow(image_np1, cmap='gray')  
                    axs[0].scatter(x_positions1, y_positions1, facecolors='none', edgecolors='red', s=50)
                    axs[0].set_title('First Image')  
                    axs[1].imshow(image_np2, cmap='gray') 
                    axs[1].scatter(x_positions2, y_positions2, facecolors='none', edgecolors='red', s=50)
                    axs[1].set_title('Second Image') 
                    plt.tight_layout()
                    plt.show()  
                
                # tracking
                print('------------ Particle Tracking--------------')
                size = (np.max(pos1[:, 0])-np.min(pos1[:, 0]))*(np.max(pos1[:, 1])-np.min(pos1[:, 1]))
                scale = np.sqrt((pos1.shape[0]/size)/(512/1800))
                pos1 = pos1 * scale
                pos2 = pos2 * scale

                pc1 = torch.from_numpy(pos1).unsqueeze(0).float()
                pc2 = torch.from_numpy(pos2).unsqueeze(0).float()
                scale = torch.from_numpy(np.array(scale))
                
                batch_data = {"sequence": [pc1, pc2]}   
                for key in batch_data.keys():
                    batch_data[key] = [d.to(device) for d in batch_data[key]]
                with torch.no_grad():
                    tracker.eval()
                    est_flow, flow_mask = tracker(batch_data['sequence'], args.nb_iter)
                
                trajectories = (est_flow != 0).any(dim=2).sum(dim=1)
                print('Number of trajectories tracked:  ', trajectories.item())
                
                est_flow = est_flow / scale
                pc1 = pc1 / scale
                pc2 = pc2 / scale
                batch_data = {"sequence": [pc1, pc2]}   
                # plot tracking results
                if args.result_plot == 1:
                    flow_pic_test(est_flow, batch_data, image_np1, x_positions1, y_positions1)
                
                pc1 = pc1.cpu().numpy()
                pc2 = pc2.cpu().numpy()
                est_flow = est_flow.cpu().numpy()
                
                data_dict = {'pc1': pc1, 'pc2': pc2, 'est_flow': est_flow, 'scale':args.scale_image}
                i_str = str(i).zfill(4)
                savemat(f'{save_path}/result_{i_str}.mat', data_dict)
                
if __name__ == '__main__':
    main()







