import os
from utils import load_h5_file, augmentation, plot_xyz, plot_pcds, pc_normalize
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import open3d


class ShapeNetDataset(Dataset):
    def __init__(self, data_path, point_class='plane', mode='train', scaling=None, rotation=False, mirror_prob=None):

        self.mode = mode
        self.partial_list = []
        self.target_list = []
        self.rotation = rotation
        self.mirror_prob = mirror_prob
        self.scaling = scaling

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

        point[:, 0:3] = pc_normalize(point[:, 0:3])
        target[:, 0:3] = pc_normalize(target[:, 0:3])
        point, target = augmentation(point, target, self.scaling, self.rotation, self.mirror_prob)

        return torch.Tensor(point.T), torch.Tensor(target.T)

# Kitti contains only test data
# normalization
class KittiDataset(Dataset):
    def __init__(self, data_path):

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
    data_dir_train = './shapenet'

    # train_dataset = PointDataset(data_dir_train, 'train', scaling=2, rotation=True, mirror_prob=0.4)

    train_dataset = ShapeNetDataset(data_dir_train, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    for input, target in train_loader:
        print(input)
        print(target)
        plot_xyz(input[0], save_path='./plot_xyz_trn.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        import pdb;pdb.set_trace()
