#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


class FluidflowDataset2D(Dataset):
    def __init__(self, npoints=2048, root='data/PTVflow2D', 
                 partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.root = root
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        #self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######
        self.datapath.sort()
        print(self.partition, ': ', len(self.datapath))

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                flow = data['flow'].astype('float32')

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

        if self.partition == 'train':
            pos1 = pos1[:self.npoints, 0:2]
            pos2 = pos2[:self.npoints, 0:2]
            flow = flow[:self.npoints, 0:2]

        else:
            pos1 = pos1[:self.npoints, 0:2]
            pos2 = pos2[:self.npoints, 0:2]
            flow = flow[:self.npoints, 0:2]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, flow

    def __len__(self):
        return len(self.datapath)


class FluidflowDataset2D_test(Dataset):
    def __init__(self, npoints=2048, root='data/PTVflow2D',
                 partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.root = root
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000

        self.datapath.sort()
        print(self.partition, ': ', len(self.datapath))

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                flow = data['flow'].astype('float32')

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

        if self.partition == 'train':
            pos1 = pos1[:self.npoints, 0:2]
            pos2 = pos2[:self.npoints, 0:2]
            flow = flow[:self.npoints, 0:2]

        else:
            pos1 = pos1[:self.npoints, 0:2]
            pos2 = pos2[:self.npoints, 0:2]
            flow = flow[:self.npoints, 0:2]
            #scale
            n_all, _ = pos1.shape
            size = (np.max(pos1[:, 0])-np.min(pos1[:, 0]))*(np.max(pos1[:, 1])-np.min(pos1[:, 1]))
            scale = np.sqrt((n_all/size)/(512/1800))
            pos1 = pos1 * scale
            pos2 = pos2 * scale
            flow = flow * scale

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, flow

    def __len__(self):
        return len(self.datapath)


# if __name__ == '__main__':
#     train = ModelNet40(1024)
#     test = ModelNet40(1024, 'test')
#     for data in train:
#         print(data[0].shape)
#         break
