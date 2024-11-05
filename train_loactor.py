import numpy as np
import torch
import torch.nn as nn
import os
import glob
import scipy.io as scio
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import math
from PIL import Image
import argparse
from model.locator import FCNN,RMSE_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='index of gpu used')
    parser.add_argument('--name', type=str, default='locator',
                        help='name of experiment')
    parser.add_argument('--input_path_ckpt', type=str, default=None,
                        help='path of already trained checkpoint')
    parser.add_argument('--recover', type=eval, default=False,
                        help='Wether to load an existing checkpoint')

    parser.add_argument('--output_dir_ckpt', type=str, default='./checkpoint/',
                        help='output directory of checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/particle_image/',
                        help='dataset for train')

    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--init_lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--reduce_factor', default=0.5, type=float,
                        help='reduce factor of ReduceLROnPlateau scheme')
    parser.add_argument('--patience_level', default=20, type=int,
                        help='patience level of ReduceLROnPlateau scheme')
    parser.add_argument('--min_lr', default=1e-8, type=float,
                        help='minimum learning rate')
    args = parser.parse_args()
    print('args parsed')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    train(args)


def data_read(images_pairs):
    images_split = []
    label = []
    for img_path in images_pairs:
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        img = img / img.max()
        target_path = img_path.replace('Img', 'ImgTarget')
        target_img = np.array(Image.open(target_path).convert('L'), dtype=np.float32)
        particle_path = img_path.replace('Img', 'ImgData')
        particle_path = particle_path.replace('tif', 'mat')
        particle_mat = scio.loadmat(particle_path)
        particle = particle_mat['Particle']
        number = particle['number'][0,0]
        position = particle['position'][0,0]
        intensity = particle['intensity'][0,0]
        diameter = particle['diameter'][0,0]

        position_int = np.round(position) - 1
        position_int = position_int.astype(int)

        img_padded = np.pad(img, pad_width=3, mode='constant', constant_values=0)
        particle_in_img = np.where(target_img == 255.0)
        y_positions = particle_in_img[0]
        x_positions = particle_in_img[1]

        patches = extract_patches_no_loop(img_padded, y_positions + 3, x_positions + 3)

        combined = np.stack([y_positions, x_positions], axis=0)

        matches = (position_int[:, :, None] == combined[:, None, :]).all(axis=0)
        indices = np.argmax(matches, axis=0)

        label_pos = position[:, indices]  - combined - 1
        label_dia = diameter[:, indices]
        label_in = intensity[:, indices]

        label_i = np.vstack((label_pos, label_dia, label_in))

        images_split.append(patches)
        label.append(label_i)
        
    return label, images_split

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

def train(args):
    device = 'cuda:' + str(args.gpu)
    image_path = args.data_path + 'Img/Train_*.tif'
    save_path = args.output_dir_ckpt + args.name + '/'
    log_file_path = save_path + 'training_log.txt'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images_pairs = glob.glob(image_path)
    label, images_split = data_read(images_pairs)

    label_all = np.concatenate(label, axis=1)
    images_split_all = np.concatenate(images_split, axis=0)

    label_all = torch.from_numpy(label_all).to(torch.float32).permute(1, 0)
    images_split_all = torch.from_numpy(images_split_all).to(torch.float32)

    train_images = images_split_all.unsqueeze(1)
    train_label_all = label_all

    image_path = args.data_path + '/Img/Val*.tif'
    images_pairs = glob.glob(image_path)
    label, images_split = data_read(images_pairs)

    label_all = np.concatenate(label, axis=1)
    images_split_all = np.concatenate(images_split, axis=0)

    label_all = torch.from_numpy(label_all).to(torch.float32).permute(1, 0)
    images_split_all = torch.from_numpy(images_split_all).to(torch.float32)

    test_images = images_split_all.unsqueeze(1)
    test_label_all = label_all

    model = FCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.reduce_factor, patience=args.patience_level, min_lr=args.min_lr)

    train_data = TensorDataset(train_images, train_label_all)
    test_data = TensorDataset(test_images, test_label_all)

    train_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    start_epoch = 0
    lowest_epoch_test_loss = 1000

    for epoch in range(start_epoch, args.epochs, 1):
        model.train()
        epoch_train_loss, epoch_test_loss = [],[]
        train_loader_len = int(math.ceil(len(train_data)))
        train_pbar = tqdm(enumerate(train_data), total=train_loader_len,
                          desc='Epoch: [' + str(epoch + 1) + '/' + str(args.epochs) + '] Training', position=0, leave=False)
        start_time = time.time()
        # Training
        with torch.set_grad_enabled(True):
            for i, sample_batched in train_pbar:
                images, images_traget = sample_batched

                images = images.to(device)
                images_traget = images_traget[:,0:2].to(device)

                pred_images = model(images)

                training_loss = RMSE_loss(pred_images, images_traget)

                optimizer.zero_grad()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                epoch_train_loss.append(training_loss)
                training_loss.backward()
                optimizer.step()

                train_pbar.set_postfix_str(
                    'loss: ' + "{:10.4f}".format(training_loss))

        epoch_train_loss = torch.mean(torch.stack(epoch_train_loss)).item()

        # Validation
        with torch.set_grad_enabled(False):
            # set evaluation mode
            model.eval()

            val_loader_len = int(math.ceil(len(test_data)))
            val_pbar = tqdm(enumerate(test_data), total=val_loader_len,
                            desc='Epoch: [' + str(epoch + 1) + '/' + str(args.epochs) + '] Validation', position=1, leave=False)

            for i, sample_batched in val_pbar:
                images, images_traget = sample_batched

                images = images.to(device)
                images_traget = images_traget[:,0:2].to(device)
                pred_images = model(images)

                test_loss = RMSE_loss(pred_images, images_traget)

                epoch_test_loss.append(test_loss)

                val_pbar.set_postfix_str(
                    'loss: ' + "{:10.4f}".format(test_loss))

        epoch_test_loss = torch.mean(torch.stack(epoch_test_loss)).item()

        scheduler.step(epoch_test_loss)
        torch.cuda.empty_cache()
        end_time = time.time()

        times = end_time - start_time

        print('Epoch: ', epoch, '  time: ', f'{times:.5f}', '   training loss: ', f'{epoch_train_loss:.8f}', '   validation loss: ', \
              f'{epoch_test_loss:.8f}', '  lr: ', f'{optimizer.param_groups[0]["lr"]:.5f}', flush=True)

        if (epoch_test_loss < lowest_epoch_test_loss):
            lowest_epoch_test_loss = epoch_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path + str(epoch) + '_ckpt.tar')

        with open(log_file_path, 'a') as log_file:
            log_file.write('Epoch: ' + str(epoch) + '   time: ' + f'{times:.5f}'+
                           '   training loss: ' + f'{epoch_train_loss:.5f}' + '   validation loss: ' + f'{epoch_test_loss:.5f}' +
                           '   lr: ' + str(optimizer.__getattribute__('param_groups')[0]['lr']) + '\n')

if __name__ == '__main__':
    main()