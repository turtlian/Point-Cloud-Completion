import torch
import numpy as np
import time
import argparse
import os
from utils import pc_normalize,load_h5_file,plot_xyz
import pickle
import h5py

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


def emd_aug(pc1,pc2,mode,lambda_,args):
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
    if mode=='partial':
        lambda_ = np.random.beta(1,1)
    emd_pc = (1-lambda_)*pc1 + (lambda_)*pc2
    return emd_pc,lambda_


def main():
    start = time.time()

    # step1. load data
    print('=== Load Data ===')
    data_path = os.path.join('/daintlab/data/shapenet/train.list')
    partial_list = []
    target_list = []
    emd_partial_list = []
    emd_target_list = []
    
    with open(data_path, 'r') as f:
        for line in f:
            partial = os.path.join('/daintlab/data/shapenet/train/partial', line.rstrip() + '.h5')
            target = os.path.join('/daintlab/data/shapenet/train/gt', line.rstrip() + '.h5')
            partial_list.append(pc_normalize(load_h5_file(partial)[:,0:3])) # (2048,3) * 28974
            target_list.append(pc_normalize(load_h5_file(target)[:,0:3])) # (2048,3) * 28974
           
    
    # step2. Euclidean sort (it dosen't take long time : 3minutes)
    print('=== Euclidean Sort ===')
    for i in range(len(partial_list)):
        partial_list[i] = euclidean_sort(partial_list[i])
        target_list[i] = euclidean_sort(target_list[i])
        if (i+1)%100==0:
            print("data / total : {} / {}".format(i+1,len(partial_list)-1))

    # step3. emd_aug (it may takes long time.. : 1hour+@)
    print("=== EMD Augmentation ===")
    for i in range(len(partial_list)-1):
        emd_pc_partial,lambda_ = emd_aug(partial_list[i],partial_list[i+1],'partial',0,args)
        emd_pc_target,_ = emd_aug(target_list[i],target_list[i+1],'target',lambda_,args)
        emd_partial_list.append(emd_pc_partial)
        emd_target_list.append(emd_pc_target)
        if (i+1)%100==0:
            print("data / total : {} / {}".format(i+1,len(partial_list)-1))

    #step4. save
    print("=== Now On Saving... ===")

    for i in range(len(emd_partial_list)):
        hf = h5py.File('./emd_mixup_data/partial/'+str(i+1)+'.h5', 'w')
        h = hf.create_dataset('data', data=emd_partial_list[i])
        hf.close()
    
    for i in range(len(emd_target_list)):
        hf = h5py.File('./emd_mixup_data/target/'+str(i+1)+'.h5', 'w')
        h = hf.create_dataset('data', data=emd_target_list[i])
        hf.close()


    # step5. save visualized plots
    k = 15000
    for i in range(10):
        # i+9100 to save airplane
        point_emd = emd_partial_list[i+k]
        point = partial_list[i+k]
        target = emd_target_list[i+k]
    
        point_emd = torch.Tensor(point_emd.T)
        point = torch.Tensor(point.T)
        target = torch.Tensor(target.T)
        plot_xyz(point, save_path='./emd_mixup_outcome/plot_xyz_origin_trn_input_'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        plot_xyz(point_emd, save_path='./emd_mixup_outcome/plot_xyz_emd_trn_input_'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        plot_xyz(target, save_path='./emd_mixup_outcome/plot_xyz_emd_trn_target_'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))


    # final. summary
    end = time.time()
    print('EMD augmentation compeleted. Time duration : {0} | Length of new data : {1}'.
          format(end-start,len(emd_partial_list)))



'''
Code bellow shows mixup results by changing its ratio.
'''
# def euclidean_sort(pc):
#     '''
#     This function returns sorted tensor by euclidean distance.
#     input : 2048by 3 tensor
#     output : 2048by 3 euclidean sorted tensor
#     '''
#     p2_norm = np.sqrt(np.sum(pc**2,axis=1))
#     idx = sorted(range(len(p2_norm)), key=lambda k: p2_norm[k])
#     sorted_pc = np.copy(pc)
#     for i in range(2048):
#         sorted_pc[i] = pc[idx[i]]
#     return sorted_pc


# def emd_aug(pc1,pc2,k,args):
#     #beta = [0.1*i for i in range(11)]
#     '''
#     This function combines two PC tensor with EMD algorithm.
#     It originaly needs 2048! calculation (It is far bigger than 10^82 which is the total number of atom on universe)
#     The function makes EMD augmented PC within 2048*8*args.theta calcuation (theta default : 1)
#     input : pc1, pc2 (2048by 3), args.theta
#     output : merged(EMD_augmented) pc
#     '''
#     research = 8*args.theta
#     temp_points = [1,1,1]
#     temp_pc2 = np.concatenate((np.copy(pc2),
#                                np.array([temp_points for i in range(research)]))) # Fix pc1 and only pc2 will be sorted by minimum EMD in finite time 
#     for i in range(2048):
#         if k ==0:
#             temp1 = np.tile(pc1[1],(research,1)) # copy pc1[i] 8 times
#             temp2 = temp_pc2[i:i+research]
#             temp3 = np.sqrt(np.sum((temp1-temp2)**2,axis=1)) # euclidean distance
#             idx = i + np.argmin(temp3)
#             pc2[i] = temp_pc2[idx]
#     lambda_ = np.random.beta(1,1)
#     emd_pc = (1-0.1*k)*pc1 + (0.1*k)*pc2
#     print((1-0.1*k),(0.1*k))
#     return emd_pc


# def main():
#     start = time.time()

#     # step1. load data
#     print('=== Load Data ===')
#     data_path = os.path.join('/daintlab/data/shapenet/train.list')
#     partial_list = []
#     target_list = []
#     emd_partial_list = []
#     emd_target_list = []
#     c = 0
#     with open(data_path, 'r') as f:
#         for line in f:
#             if (c == 25000) or (c==13000):
#                 partial = os.path.join('/daintlab/data/shapenet/train/partial', line.rstrip() + '.h5')
#                 target = os.path.join('/daintlab/data/shapenet/train/gt', line.rstrip() + '.h5')
#                 partial_list.append(pc_normalize(load_h5_file(partial)[:,0:3])) # (2048,3) * 28974
#                 target_list.append(pc_normalize(load_h5_file(target)[:,0:3])) # (2048,3) * 28974
#                 c+=1
#                 if c==25001:
#                     break
#             else:
#                 c+=1
           
    
#     # step2. Euclidean sort (it dosen't take long time : 3minutes)
#     print('=== Euclidean Sort ===')
#     for i in range(len(partial_list)):
#         partial_list[i] = euclidean_sort(partial_list[i])
#         target_list[i] = euclidean_sort(target_list[i])
#         if (i+1)%100==0:
#             print("data / total : {} / {}".format(i+1,len(partial_list)-1))

#     # step3. emd_aug (it may takes long time.. : 1hour+@)
#     print("=== EMD Augmentation ===")
#     for i in range(11):
#         emd_pc_partial = emd_aug(partial_list[0],partial_list[1],i,args)
#         emd_pc_target = emd_aug(target_list[0],target_list[1],i,args)
#         emd_partial_list.append(emd_pc_partial)
#         emd_target_list.append(emd_pc_target)
#         if (i+1)%100==0:
#             print("data / total : {} / {}".format(i+1,len(partial_list)-1))


#     print(len(emd_partial_list))
#     # step5. save visualized plots
#     k = 0
#     for i in range(11):
#         # i+9100 to save airplane
#         point_emd = emd_partial_list[i+k]
#         #point = partial_list[i+k]
#         #target = emd_target_list[i+k]
    
#         point_emd = torch.Tensor(point_emd.T)
#         #point = torch.Tensor(point.T)
#         #target = torch.Tensor(target.T)
#         #plot_xyz(point, save_path='./emd_mixup_ratio/plot_xyz_origin_trn_input_'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
#         plot_xyz(point_emd, save_path='./emd_mixup_ratio/plot_xyz_emd_trn_input_'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
#         #plot_xyz(target, save_path='./emd_mixup_ratio/plot_xyz_emd_trn_target_'+str(i)+'.png', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))


#     # final. summary
#     end = time.time()
#     print('EMD augmentation compeleted. Time duration : {0} | Length of new data : {1}'.
#           format(end-start,len(emd_partial_list)))

if __name__ == '__main__':
    main()

    """To load data, follow bellow code. type : list[np.array,np.array,....,np.array]"""
    # f = h5py.File('./emd_mixup_data/partial/1.h5', 'r')
    # dset = f['data']
    # data = dset[:]
    # print(data) # which is a 2048 by 3 np.ndarray