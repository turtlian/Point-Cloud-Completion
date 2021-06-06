import torch
import numpy as np
import time
import argparse
import os
from utils import *

parser = argparse.ArgumentParser(description='emd_augmentation')
parser.add_argument('--theta', default=1, type=int,
                    help='EMD searches 8*theta of euclidean distance (default : 1, max : 32)')
parser.add_argument('--class', default='all', type=str,
                    help='class to mix (default : all, there are 8 classes; car, plane,...)')                    
args = parser.parse_args()


def euclidean_sort(pc):
    '''
    This function returns sorted tensor by euclidean distance.
    input : 2048by 3 tensor
    output : 2048by 3 euclidean sorted tensor
    '''
    p2_norm = np.sqrt(np.sum(pc**2,axis=1))
    idx = sorted(range(len(p2_norm)), key=lambda k: p2_norm[k])
    sorted_pc = np.copy(pc)
    for i in range(2048):
        sorted_pc[i] = pc[idx[i]]
    return sorted_pc


def emd_aug(pc1,pc2,args):
    '''
    This function combines two PC tensor with EMD algorithm.
    It originaly needs 2048! calculation (It is far bigger than 10^82 which is the total number of atom on universe)
    The function makes EMD augmented PC within 2048*8*args.theta calcuation (theta default : 1)
    input : pc1, pc2 (2048by 3), args.theta
    output : merged(EMD_augmented) pc
    '''
    research = 8*args.theta
    temp_points = [1,1,1]
    temp_pc2 = np.concatenate((np.copy(pc2),
                               np.array([temp_points for i in range(research)]))) # Fix pc1 and only pc2 will be sorted by minimum EMD in finite time 
    for i in range(2048):
        temp1 = np.tile(pc1[1],(research,1)) # copy pc1[i] 8 times
        temp2 = temp_pc2[i:i+research]
        temp3 = np.sqrt(np.sum((temp1-temp2)**2,axis=1)) # euclidean distance
        idx = i + np.argmin(temp3)
        pc2[i] = temp_pc2[idx]
    lambda_ = np.random.beta(1,1)
    emd_pc = (1-lambda_)*pc1 + (lambda_)*pc2
    return emd_pc


def main():
    start = time.time()

    # step1. load data
    data_path = os.path.join('/daintlab/data/shapenet/train.list')
    partial_list = []
    target_list = []
    emd_partial_list = []
    emd_target_list = []
    c =0 ###
    with open(data_path, 'r') as f:
        for line in f:
            partial = os.path.join('/daintlab/data/shapenet/train/partial', line.rstrip() + '.h5')
            target = os.path.join('/daintlab/data/shapenet/train/gt', line.rstrip() + '.h5')
            partial_list.append(pc_normalize(load_h5_file(partial)[:,0:3])) # (2048,3) * 28974
            target_list.append(pc_normalize(load_h5_file(target)[:,0:3])) # (2048,3) * 28974
            if c==10: ###
                break ###
            c+=1 ###
    
    # step2. Euclidean sort
    for i in range(len(partial_list)):
        partial_list[i] = euclidean_sort(partial_list[i])
        target_list[i] = euclidean_sort(target_list[i])

    # step3. emd_aug (it may takes long time..)
    print("=== EMD Augmentation ===")
    for i in range(len(partial_list)-1):
        emd_pc_partial = emd_aug(partial_list[i],partial_list[i+1],args)
        emd_pc_target = emd_aug(target_list[i],target_list[i+1],args)
        emd_partial_list.append(emd_pc_partial)
        emd_target_list.append(emd_pc_target)
        if (i+1)%100==0:
            print("data / total : {} / {}".format(i+1,len(partial_list-1)))

    # step4. save

    # step5. save visualized plots
    for i in range(10):
        point = emd_partial_list[i]
        target = emd_target_list[i]
        #point[:, 0:3] = pc_normalize(point[:, 0:3])
        #target[:,0:3] = pc_normalize(target[:,0:3])
    
        point, target = augmentation(point, target, None, False, None)
        point = torch.Tensor(point.T)
        plot_xyz(point, save_path='./plot_xyz_emd_trn'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))

    ### ===== ###
    end = time.time()
    print('EMD augmentation compeleted. Time duration : {0} | Length of new data : {1}'.
          format(end-start,len(emd_partial_list)))


if __name__ == '__main__':
    main()