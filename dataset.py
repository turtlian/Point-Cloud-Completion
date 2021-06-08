import os
from utils import load_h5_file, augmentation, resample_pcd, plot_xyz, plot_pcds, pc_normalize, mix_up
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import open3d
import numpy as np

class ShapeNetDataset(Dataset):
    def __init__(self, data_path, point_class='all', mode='train', scaling=None, rotation=False, mirror_prob=None,
                 crop_prob=None, num_coarse = 1024, num_dense = 16384):

        self.mode = mode
        self.partial_list = []
        self.target_list = []
        self.rotation = rotation
        self.mirror_prob = mirror_prob
        self.scaling = scaling
        self.crop_prob = crop_prob
        self.num_coarse = num_coarse
        self.num_dense = num_dense

        if point_class != 'all':
            self.data_path = data_path
            classmap = pd.read_csv(data_path+'/synsetoffset2category.txt', header=None, sep = '\t')
            class_dict = {}
            for i in range(classmap.shape[0]):
                class_dict[classmap[0][i]] = str(classmap[1][i]).zfill(8)

            self.partial_path_list = os.path.join(data_path, mode, 'partial', class_dict[point_class])
            self.partial_path_list = sorted([os.path.join(self.partial_path_list, k) for k in os.listdir(self.partial_path_list)])
            self.target_path_list = os.path.join(data_path, mode, 'gt', class_dict[point_class])
            self.target_path_list = sorted([os.path.join(self.target_path_list, k) for k in os.listdir(self.target_path_list)])

            for i, path in enumerate(self.partial_path_list):
                self.partial_list.append(load_h5_file(path))
                self.target_list.append(load_h5_file(self.target_path_list[i]))
        else :
            self.data_path = os.path.join(data_path, self.mode + '.list')
            with open(self.data_path, 'r') as f:
                for line in f:
                    partial = os.path.join(data_path, self.mode, 'partial', line.rstrip() + '.h5')
                    target = os.path.join(data_path, self.mode, 'gt', line.rstrip() + '.h5')
                    self.partial_list.append(load_h5_file(partial))
                    self.target_list.append(load_h5_file(target))

    def __len__(self):
        return len(self.partial_list)

    def __getitem__(self, idx):
        point = self.partial_list[idx]
        target = self.target_list[idx]
        # augmentation
        point, target = augmentation(point, target, self.scaling, self.rotation, self.mirror_prob, self.crop_prob)
        # normalizing
        point[:, 0:3] = pc_normalize(point[:, 0:3])
        target[:, 0:3] = pc_normalize(target[:, 0:3])
        # sub sampling
        choice = np.random.choice(len(point), self.num_coarse, replace=True)
        coarse_gt = target[choice, :]
        dense_gt = resample_pcd(target, self.num_dense)

        return torch.Tensor(point.T), torch.Tensor(target.T), torch.Tensor(coarse_gt.T), torch.Tensor(dense_gt.T)


# Kitti contains only test data
class KittiDataset(Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        self.data_list = os.listdir(os.path.join(self.data_path, 'cars'))
        self.point_list=[]
        for i in self.data_list:
            point = open3d.io.read_point_cloud(os.path.join(self.data_path,'cars',i)).points
            self.point_list.append(np.asarray(point))

    def __len__(self):
        return len(self.point_list)

    def __getitem__(self, idx):
        point = self.point_list[idx]
        point[:, 0:3] = pc_normalize(point[:, 0:3])

        return torch.Tensor(point.T)
