import h5py
import numpy as np
import transforms3d
import random
import math
from matplotlib import pyplot as plt
from collections import Iterable
import torch
import random

# data
def load_h5_file(path):
    '''Load point cloud data from h5 file'''
    f = h5py.File(path, 'r')
    point_data = np.array(f['data'], dtype = np.float64)
    f.close()

    return point_data

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def augmentation(point, target, scale, rotation, mirror_prob, crop_prob):
    '''https://github.com/matthew-brett/transforms3d'''
    transform_matrix = transforms3d.zooms.zfdir2mat(1)

    if scale is not None:
        scaling_num = random.uniform(1 / scale, scale)
        transform_matrix = np.dot(transforms3d.zooms.zfdir2mat(scaling_num), transform_matrix)
    if rotation:
        angle = random.uniform(0, 2*math.pi)
        transform_matrix = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), transform_matrix) #fix y-axis
    if mirror_prob is not None:
        # flip x&z, not z
        if random.random() < mirror_prob/2:
            transform_matrix = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), transform_matrix)
        if random.random() < mirror_prob/2:
            transform_matrix = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), transform_matrix)
        if random.random() < mirror_prob/2:
            transform_matrix = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), transform_matrix)
    if crop_prob is not None and np.random.rand(1) < crop_prob:
        point_c = pc_normalize(point[:, 0:3])
        target_c = pc_normalize(target[:, 0:3])
        voxel_idx = np.random.randint(1, 5)
        if voxel_idx == 1:
            point_c=point_c[point_c[:, 0] >= 0];point_c = point_c[point_c[:, 2] >= 0]
            target_c = target_c[target_c[:, 0] >= 0];target_c = target_c[target_c[:, 2] >= 0]
        elif voxel_idx == 2:
            point_c = point_c[point_c[:, 0] >= 0];point_c = point_c[point_c[:, 2] <= 0]
            target_c = target_c[target_c[:, 0] >= 0];target_c = target_c[target_c[:, 2] <= 0]
        elif voxel_idx ==3:
            point_c = point_c[point_c[:, 0] <= 0];point_c = point_c[point_c[:, 2] >= 0]
            target_c = target_c[target_c[:, 0] <= 0];target_c = target_c[target_c[:, 2] >= 0]
        elif voxel_idx ==4:
            point_c = point_c[point_c[:, 0] <= 0];point_c = point_c[point_c[:, 2] <= 0]
            target_c = target_c[target_c[:, 0] <= 0];target_c = target_c[target_c[:, 2] <= 0]
        if point_c.shape[0] > 200 and point_c.shape[0] <= 1024:
            try:
                point = resample_pcd(point_c, 2048)
                target = resample_pcd(target_c, 2048)
            except ValueError:
                print('')

    return np.dot(point, transform_matrix), np.dot(target, transform_matrix)


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])

    return pcd[idx[:n]]

def mix_up(partial, coarsegt, densegt, mixup, alpha):
    if mixup == 'naive':
        '''
        2개의 input point cloud를 불러옵니다. s1,s2
        s1에서 2048*gamma만큼의 points를 random하게 가져오고 
        s2에서 2048*(1-gamma)만큼의 points를 random하게 가져옵니다
        둘을 concat
        '''
        # shuffle idx
        rand_idx = torch.randperm(partial.size(0))
        partial_shuffle = partial[rand_idx]
        coarsegt_shuffle = coarsegt[rand_idx]
        densegt_shuffle = densegt[rand_idx]

        # mixup : 1st+2nd, 2nd+3rd, 3rd+4th.....
        gamma = np.random.beta(alpha,alpha) # 0~1
        split = int(2048*gamma)
        partial = torch.cat([partial[:, :, 0:split],partial_shuffle[:, :, split:]],dim=2)
        coarsegt = torch.cat([coarsegt[:, :, 0:split], coarsegt_shuffle[:, :, split:]], dim=2)
        densegt = torch.cat([densegt[:, :, 0:split], densegt_shuffle[:, :, split:]], dim=2)

    elif mixup == 'emd':
        print('EEEEEMMMMMMMDDDD!!!')

    return partial, coarsegt, densegt


# visualization
'''https://github.com/lynetcha/completion3d/blob/1dc8ffac02c4ec49afb33c41f13dd5f90abdf5b7/shared/vis.py'''

def plot_xyz(xyz, zdir='y', cmap='Reds', xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3), save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    elev = 30
    azim = -45
    ax.view_init(elev, azim)
    xyz = xyz.T
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], c=xyz[:,0], s=20, zdir=zdir, cmap=cmap, vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_pcds(pcds, titles, use_color=[],color=None, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3), save_path=None):
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            if color is None or not use_color[j]:
                pcd = pcd.T
                clr = pcd[:, 0]

            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=clr, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


# logging
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0  # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val != None:  # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val ** 2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2 / self.count - self.avg ** 2)
        else:
            pass


class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try:
            return len(self.read())
        except:
            return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.', v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def draw_curve(work_dir, train_logger, test_logger):
        train_logger = train_logger.read()
        test_logger = test_logger.read()
        epoch, train_loss = zip(*train_logger)
        epoch,test_loss = zip(*test_logger)

        plt.plot(epoch, train_loss, color='blue', label="Train Loss")
        plt.plot(epoch, test_loss, color='red', label="Test Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(work_dir + '/loss_curve.png')
        plt.close()
