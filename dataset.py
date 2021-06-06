import os
from utils import load_h5_file, augmentation, resample_pcd, plot_xyz, plot_pcds, pc_normalize, mix_up
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import open3d
import numpy as np

class ShapeNetDataset(Dataset):
    def __init__(self, data_path, point_class='plane', mode='train', scaling=None, rotation=False, mirror_prob=None,
                 num_coarse = 1024, num_dense = 16384, mixup=None):

        self.mode = mode
        self.partial_list = []
        self.target_list = []
        self.rotation = rotation
        self.mirror_prob = mirror_prob
        self.scaling = scaling
        self.num_coarse = num_coarse
        self.num_dense = num_dense

        if point_class != 'all': # 특정 클래스에서만 가져오기
            self.data_path = data_path
            classmap = pd.read_csv(data_path+'/synsetoffset2category.txt', header=None, sep = '\t')
            class_dict = {}
            for i in range(classmap.shape[0]):
                class_dict[classmap[0][i]] = str(classmap[1][i]).zfill(8)

            self.partial_path_list = os.path.join(data_path, mode, 'partial', class_dict[point_class]) # partial data path
            self.partial_path_list = sorted([os.path.join(self.partial_path_list, k) for k in os.listdir(self.partial_path_list)]) # 해당 path에 있는 모든 데이터의 dir
            self.target_path_list = os.path.join(data_path, mode, 'gt', class_dict[point_class]) # gt data path
            self.target_path_list = sorted([os.path.join(self.target_path_list, k) for k in os.listdir(self.target_path_list)]) # 해당 path에 있는 모든 데이터의 dir
   
            for i, path in enumerate(self.partial_path_list): # 각각의 list에 partical과 target을 넣어줌. 각각은 (2048,3) 
                self.partial_list.append(load_h5_file(path))
                self.target_list.append(load_h5_file(self.target_path_list[i]))

        else : # point_class == 'all'
            self.data_path = os.path.join(data_path, self.mode + '.list')
            self.partial_path_list = []
            self.target_path_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    partial = os.path.join(data_path, self.mode, 'partial', line.rstrip() + '.h5')
                    target = os.path.join(data_path, self.mode, 'gt', line.rstrip() + '.h5')
                    self.partial_path_list.append(partial)
                    self.target_path_list.append(target)
                    self.partial_list.append(load_h5_file(partial))
                    self.target_list.append(load_h5_file(target))
            if mixup != None:
                self.partial_list, self.target_list = mix_up(self.partial_path_list,
                                                             self.target_path_list,
                                                             mixup)
        
    def __len__(self):
        return len(self.partial_list)

    def __getitem__(self, idx):
        point = self.partial_list[idx]
        target = self.target_list[idx]
        print('pp',point.shape)

        point[:, 0:3] = pc_normalize(point[:, 0:3])
        target[:, 0:3] = pc_normalize(target[:, 0:3])

        choice = np.random.choice(len(point), self.num_coarse, replace = True) # scalar
        coarse_gt = target[choice, : ]
        dense_gt = resample_pcd(target, self.num_dense)
        point, target = augmentation(point, target, self.scaling, self.rotation, self.mirror_prob)

        return torch.Tensor(point.T), torch.Tensor(target.T), torch.Tensor(coarse_gt.T), torch.Tensor(dense_gt.T)


# Kitti contains only test data
# normalization
class KittiDataset(Dataset):
    def __init__(self, data_path,mixup=None):

        self.data_path = data_path
        self.data_list = os.listdir(os.path.join(self.data_path, 'cars'))
        self.point_list=[]
        for i in self.data_list:
            point = open3d.io.read_point_cloud(os.path.join(self.data_path,'cars',i)).points
            self.point_list.append(torch.Tensor(point))

    def __len__(self):
        return len(self.point_list)

    def __getitem__(self, idx):
        point = self.point_list[idx]
        point[:, 0:3] = pc_normalize(point[:, 0:3])

        return point.T


if __name__ == '__main__':
    data_dir_train = '/daintlab/data/shapenet'

    # train_dataset = PointDataset(data_dir_train, 'train', scaling=2, rotation=True, mirror_prob=0.4)

    train_dataset = ShapeNetDataset(data_dir_train, point_class='all',mode='train',mixup='naive')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    for i,data in enumerate(train_loader):
        input,target,coarse_gt,dense_gt = data
        print(input.shape)
        print(target.shape)
        print(coarse_gt.shape)
        print(dense_gt.shape)
        plot_xyz(input[0], save_path='./plot_xyz_trn'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        if i== 5:
            break
    # for input, target, coarse, dense in train_loader:
    #     print(input)
    #     print(target)
    #     plot_xyz(input[0], save_path='./plot_xyz_trn.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    #     import pdb;pdb.set_trace()