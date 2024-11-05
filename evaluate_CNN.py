import numpy as np
import torch
import os
import glob
import scipy.io as scio
from PIL import Image
import argparse
from model.locator import FCNN
from model.detector import UNet
import matplotlib.pyplot as plt
import time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='index of gpu used')
    parser.add_argument('--name', type=str, default='Particle_detection',
                        help='name of experiment')
    parser.add_argument('--detector_path', type=str, default='./precomputed_ckpts/CNN_detector/ckpt.tar',
                        help='path of already trained checkpoint for CNN detector (pixel level)')
    parser.add_argument('--locator_path', type=str, default='./precomputed_ckpts/CNN_locator/ckpt.tar',
                        help='path of already trained checkpoint for CNN locator (sub-pixel level)')

    parser.add_argument('--data_path', type=str, default='./data/particle_image/',
                        help='dataset for train')
    parser.add_argument('--output_dir_ckpt', type=str, default='./result/',
                        help='output directory for results')

    parser.add_argument('--result_plot', type=int, default=0,
                        choices=[0, 1],
                        help='Whether to plot the results. 0: No; 1: Yes')
    parser.add_argument('--save', type=int, default=0,
                        choices=[0, 1],
                        help='Whether to save the results. 0: No; 1: Yes')
    args = parser.parse_args()
    print('args parsed')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    train(args)


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
    half_size = patch_size // 2

    x_idx = x_positions[:, None, None] + np.arange(-half_size, half_size + 1)[None, :, None]
    y_idx = y_positions[:, None, None] + np.arange(-half_size, half_size + 1)[None, None, :]

    x_idx = np.clip(x_idx, 0, H - 1)
    y_idx = np.clip(y_idx, 0, W - 1)

    patches = img[x_idx, y_idx]

    return patches

def data_read(images_pairs):
    images = []
    position_gt = []
    for img_path in images_pairs:

        img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        img = img / img.max()

        height, width = img.shape

        new_height = (height // 8) * 8
        new_width = (width // 8) * 8

        new_height = new_height - new_height % 2
        new_width = new_width - new_width % 2

        cropped_image = img[:new_height, :new_width]
        images.append(cropped_image)

        particle_path = img_path.replace('Img', 'ImgData')
        particle_path = particle_path.replace('tif', 'mat')
        particle_mat = scio.loadmat(particle_path)
        particle = particle_mat['Particle']
        position = particle['position'][0,0] -1

        position_gt.append(position)

    return images, position_gt

def train(args):
    device = 'cuda:' + str(args.gpu)
    save_path = args.output_dir_ckpt + args.name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_path = args.data_path + '/Img/Val*.tif'
    images_pairs = glob.glob(image_path)

    images, position_gt = data_read(images_pairs)
    images = np.array(images)
    images = torch.from_numpy(images).to(torch.float32)

    test_images = images.unsqueeze(1)

    print('total number of test samples:', test_images.shape[0])

    model = UNet(in_channels=1, out_channels=1).to(device)
    locator = FCNN().to(device)

    checkpoint = torch.load(args.detector_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint_locator = torch.load(args.locator_path, map_location=device)
    locator.load_state_dict(checkpoint_locator['model_state_dict'])

    with torch.no_grad():
        model.eval()
        locator.eval()
        acc_05_mean = []
        Reliability_mean = []
        tt = 0
        for i in range(test_images.shape[0]):
            t1 = time.time()
            sample_i = i
            position_gt_i = np.array(position_gt[i])
            image = test_images[i:i + 1, ...].to(device)
            pred_images = model(image)

            pred_images = torch.round(pred_images)
            patches, y_positions, x_positions = img_split(image.squeeze(0).squeeze(0),
                                                          pred_images.squeeze(0).squeeze(0))

            patches = torch.from_numpy(patches).to(device)
            y_positions = torch.from_numpy(y_positions).to(device)
            x_positions = torch.from_numpy(x_positions).to(device)

            pred_pos = locator(patches.unsqueeze(1))

            pred_y = pred_pos[:, 0:1]
            pred_x = pred_pos[:, 1:2]
            pos_pred = torch.cat((pred_y, pred_x), dim=1) + torch.cat(
                (y_positions.unsqueeze(-1), x_positions.unsqueeze(-1)), dim=1)

            t2 = time.time()
            image_np = image.cpu().squeeze().numpy()
            pre_y_positions = pos_pred[:, 0].cpu().numpy()
            pre_x_positions = pos_pred[:, 1].cpu().numpy()

            # plot
            if args.result_plot ==1:
                plt.figure(figsize=(8, 8))
                plt.imshow(image_np, cmap='gray')
                plt.scatter(pre_x_positions, pre_y_positions, facecolors='none', edgecolors='red', s=50, label='prediction')
                plt.scatter(position_gt_i[1,:], position_gt_i[0,:], facecolors='green',  marker='+', s=50, label='ground truth')
                plt.show()

            pos_pred = pos_pred.T.cpu().numpy()
            position_gt_i = position_gt_i

            distances = np.sqrt(np.sum((position_gt_i[:, :, np.newaxis] - pos_pred[:, np.newaxis, :]) ** 2, axis=0))
            total_count = 0
            while distances.size > 0:
                nearest_in_B = np.argmin(distances, axis=1)
                nearest_in_A = np.argmin(distances, axis=0)
                mutual_nearest = [(i, j) for i, j in enumerate(nearest_in_B) if
                                  nearest_in_A[j] == i and distances[i, j] < 0.5]
                N1 = len(mutual_nearest)
                total_count += N1
                if N1 == 0:
                    break
                rows_to_remove = [i for i, _ in mutual_nearest]
                cols_to_remove = [j for _, j in mutual_nearest]
                distances = np.delete(distances, rows_to_remove, axis=0)
                distances = np.delete(distances, cols_to_remove, axis=1)

            acc_05 = total_count / position_gt_i.shape[1]
            reliability = total_count / pos_pred.shape[1]
            print(f"Sample: {sample_i} ", f"   acc(0.5 px): {acc_05:.4f}", f"   Reliability: {reliability:.4f}")
            acc_05_mean.append(acc_05)
            Reliability_mean.append(reliability)
            tt = t2 - t1 + tt

            data_dict = {'pc1': pos_pred}
            i_str = str(i).zfill(4)
            if args.save == 1:
                scio.savemat(f'{save_path}/result_{i_str}.mat', data_dict)

        print(f"Avg Acc ±0.5 px: {np.mean(acc_05_mean):.4f} ", f"   Avg Reliability: {np.mean(Reliability_mean):.4f} ")
        print('time:', tt)
if __name__ == '__main__':
    main()