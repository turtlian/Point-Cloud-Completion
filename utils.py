import h5py
import numpy as np
import transforms3d
import random
import math
from matplotlib import pyplot as plt

# data
def load_h5_file(path):
    '''Load point cloud data from h5 file'''
    f = h5py.File(path, 'r')
    point_data = np.array(f['data'], dtype = np.float64)
    f.close()

    return point_data


def augmentation(partial, target, scale, rotation, mirror_prob):
    '''https://github.com/matthew-brett/transforms3d'''
    transform_matrix = transforms3d.zooms.zfdir2mat(1)

    if scale is not None:
        scaling_num = random.uniform(1/scale, scale)
        transform_matrix = np.dot(transforms3d.zooms.zfdir2mat(scaling_num), transform_matrix)
    if rotation:
        angle = random.uniform(0, 2*math.pi)
        transform_matrix = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), transform_matrix) #fix y-axis
    if mirror_prob is not None:
        # mirroring x&z, not y
        if random.random() < mirror_prob/2:
            transform_matrix = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), transform_matrix)
        if random.random() < mirror_prob/2:
            transform_matrix = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), transform_matrix)

    return np.dot(partial, transform_matrix), np.dot(target, transform_matrix)


# visualization
'''https://github.com/lynetcha/completion3d/blob/1dc8ffac02c4ec49afb33c41f13dd5f90abdf5b7/shared/vis.py'''

def plot_xyz(xyz, zdir='y', cmap='Reds', xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3), save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    elev = 30
    azim = -45
    ax.view_init(elev, azim)
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
