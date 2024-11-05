import numpy as np
import torch
import torch.nn as nn
import os
import glob
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import math
from PIL import Image
import argparse
from model.detector import loss_metric,RMSE_loss,UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='index of gpu used')
    parser.add_argument('--name', type=str, default='detector',
                        help='name of experiment')
    parser.add_argument('--input_path_ckpt', type=str, default=None,
                        help='path of already trained checkpoint')
    parser.add_argument('--recover', type=eval, default=False,
                        help='Wether to load an existing checkpoint')

    parser.add_argument('--output_dir_ckpt', type=str, default='./checkpoint/',
                        help='output directory of checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/particle_image/',
                        help='dataset for train')

    parser.add_argument('--batch_size', default=10, type=int)
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
    images = []
    images_target = []

    for img_path in images_pairs:
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        img = img / img.max()
        images.append(img)
        target_path = img_path.replace('Img', 'ImgTarget')

        target_img = np.array(Image.open(target_path).convert('L'), dtype=np.float32)
        images_target.append(target_img)
    return images, images_target

def train(args):
    device = 'cuda:'+str(args.gpu)

    model = UNet(in_channels=1, out_channels=1).to(device)
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable parameters: ', pytorch_trainable_params)

    image_path = args.data_path + 'Img/Train_*.tif'
    save_path = args.output_dir_ckpt + args.name + '/'
    log_file_path = save_path + 'training_log.txt'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    images_pairs = glob.glob(image_path)

    images, images_target = data_read(images_pairs)
    images = np.array(images)
    images = torch.from_numpy(images).to(torch.float32)
    images_target = np.array(images_target)
    images_target = torch.from_numpy(images_target).to(torch.float32) / 255.0

    train_images = images.unsqueeze(1)
    train_images_target = images_target.unsqueeze(1)

    image_path = args.data_path + '/Img/Val*.tif'
    images_pairs = glob.glob(image_path)

    images, images_target = data_read(images_pairs)
    images = np.array(images)
    images = torch.from_numpy(images).to(torch.float32)
    images_target = np.array(images_target)
    images_target = torch.from_numpy(images_target).to(torch.float32) / 255.0

    test_images = images.unsqueeze(1)
    test_images_target = images_target.unsqueeze(1)

    print('total number for trainging samples:', train_images.shape[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.reduce_factor, patience=args.patience_level, min_lr=args.min_lr)

    train_data = TensorDataset(train_images, train_images_target)
    test_data = TensorDataset(test_images, test_images_target)

    train_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    start_epoch = 0
    lowest_epoch_test_acc_rate = 0

    if args.recover:
        checkpoint = torch.load(args.input_path_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print("recovering at epoch: ", start_epoch)

    for epoch in range(start_epoch, args.epochs, 1):
        model.train()
        epoch_train_miss_rate,epoch_test_miss_rate = [],[]
        epoch_train_acc_rate,epoch_test_acc_rate = [],[]
        epoch_train_yield_rate, epoch_test_yield_rate = [], []
        epoch_train_loss,epoch_test_loss = [],[]
        train_loader_len = int(math.ceil(len(train_data)))
        train_pbar = tqdm(enumerate(train_data), total=train_loader_len,
                          desc='Epoch: [' + str(epoch + 1) + '/' + str(args.epochs) + '] Training', position=0, leave=False)
        start_time = time.time()
        # Training
        with torch.set_grad_enabled(True):
            for i, sample_batched in train_pbar:
                images, images_traget = sample_batched

                images = images.to(device)
                images_traget = images_traget.to(device)

                pred_images = model(images)

                training_loss = RMSE_loss(pred_images, images_traget)
                metrics = loss_metric(pred_images, images_traget)

                train_miss_rate = metrics['miss_rate']
                train_acc_rate = metrics['acc_rate']
                train_yield_rate = metrics['yield_rate']
                epoch_train_miss_rate.append(train_miss_rate)
                epoch_train_acc_rate.append(train_acc_rate)
                epoch_train_yield_rate.append(train_yield_rate)

                optimizer.zero_grad()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                epoch_train_loss.append(training_loss)
                training_loss.backward()
                optimizer.step()

                train_pbar.set_postfix_str(
                    'miss_rate: ' + "{:10.4f}".format(train_miss_rate) + \
                    ' acc_rate: ' + "{:10.4f}".format(train_acc_rate) + \
                    ' yield_rate: ' + "{:10.4f}".format(train_yield_rate))

        epoch_train_loss = torch.mean(torch.stack(epoch_train_loss)).item()
        epoch_train_miss_rate = sum(epoch_train_miss_rate) / len(epoch_train_miss_rate)
        epoch_train_acc_rate = sum(epoch_train_acc_rate) / len(epoch_train_acc_rate)
        epoch_train_yield_rate = sum(epoch_train_yield_rate) / len(epoch_train_yield_rate)

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
                images_traget = images_traget.to(device)
                pred_images = model(images)

                test_loss = RMSE_loss(pred_images, images_traget)
                metrics = loss_metric(pred_images, images_traget)

                test_miss_rate = metrics['miss_rate']
                test_acc_rate = metrics['acc_rate']
                test_yield_rate = metrics['yield_rate']
                epoch_test_miss_rate.append(test_miss_rate)
                epoch_test_acc_rate.append(test_acc_rate)
                epoch_test_yield_rate.append(test_yield_rate)

                epoch_test_loss.append(test_loss)

                val_pbar.set_postfix_str(
                    'miss_rate: ' + "{:10.4f}".format(test_miss_rate) + \
                    ' acc_rate: ' + "{:10.4f}".format(test_acc_rate) + \
                    ' yield_rate: ' + "{:10.4f}".format(test_yield_rate))

        epoch_test_loss = torch.mean(torch.stack(epoch_test_loss)).item()
        epoch_test_miss_rate = sum(epoch_test_miss_rate) / len(epoch_test_miss_rate)
        epoch_test_acc_rate = sum(epoch_test_acc_rate) / len(epoch_test_acc_rate)
        epoch_test_yield_rate = sum(epoch_test_yield_rate) / len(epoch_test_yield_rate)

        scheduler.step(epoch_test_loss)
        torch.cuda.empty_cache()
        end_time = time.time()

        times = end_time - start_time

        print('Epoch: ', epoch, '  time: ', f'{times:.5f}', '   training loss: ', f'{epoch_train_loss:.8f}', '   validation loss: ', \
              f'{epoch_test_loss:.8f}', '   train miss_rate: ', f'{epoch_train_miss_rate:.5f}', '   val miss_rate: ', \
              f'{epoch_test_miss_rate:.5f}', '   train acc_rate: ', f'{epoch_train_acc_rate:.5f}', '   val acc_rate: ', \
              f'{epoch_test_acc_rate:.5f}', '   train yield_rate: ', f'{epoch_train_yield_rate:.5f}', '   val yield_rate: ', \
              f'{epoch_test_yield_rate:.5f}', '  lr: ', str(optimizer.__getattribute__('param_groups')[0]['lr']), flush=True)

        if (epoch_test_acc_rate > lowest_epoch_test_acc_rate):
            lowest_epoch_test_acc_rate = epoch_test_acc_rate
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path + str(epoch) + '_ckpt.tar')

        with open(log_file_path, 'a') as log_file:
            log_file.write('Epoch: ' + str(epoch) + '   time: ' + f'{times:.5f}'+
                           '   training loss: ' + f'{epoch_train_loss:.5f}' + '   validation loss: ' + f'{epoch_test_loss:.5f}' +
                           '   train miss_rate: ' + f'{epoch_train_miss_rate:.5f}' + '   val miss_rate: ' + f'{epoch_test_miss_rate:.5f}' +
                           '   train acc_rate: ' + f'{epoch_train_acc_rate:.5f}' + '   val acc_rate: ' + f'{epoch_test_acc_rate:.5f}' +
                           '   train yield_rate: ' + f'{epoch_train_yield_rate:.5f}' + '   val yield_rate: ' + f'{epoch_test_yield_rate:.5f}' +
                           '   lr: ' + str(optimizer.__getattribute__('param_groups')[0]['lr']) + '\n')


if __name__ == '__main__':
    main()