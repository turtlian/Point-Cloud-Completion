import dataset
from model import TopNet
import os
from utils import plot_xyz, plot_pcds
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

parser = argparse.ArgumentParser(description='visualization')
parser.add_argument('--model_path', default='./scaling_2/topnet_23.pth', type=str,
                    help='model path')
parser.add_argument('--model', default='topnet', type=str,
                    help='model name')
parser.add_argument('--embedding_dim', default=1024, type=int,
                    help='embedding size')
parser.add_argument('--data_path', default='./shapenet', type=str,
                    help='data path')
parser.add_argument('--data', default='shapenet', type=str,
                    help='dataset name')
parser.add_argument('--save_path', default='./scaling_2', type=str,
                    help='save_path')
parser.add_argument('--gpu_id', default=4, type=int,
                    help='gpu')
args = parser.parse_args()

# load dataset
if args.data == 'shapenet':
    test_dataset = dataset.ShapeNetDataset(args.data_path, mode='val')
    train_dataset = dataset.ShapeNetDataset(args.data_path, mode='train')
elif args.data == 'kitti':
    test_dataset = dataset.KittiDataset(args.data_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

# load trained model
if args.model == 'topnet':
    network = TopNet(args.embedding_dim, args.gpu_id)
network=network.cuda(args.gpu_id)
network.load_state_dict(torch.load(args.model_path))
network.eval()

# visualization
if not os.path.exists(os.path.join(args.save_path, 'target_points')):
    os.makedirs(os.path.join(args.save_path, 'target_points'))
    os.makedirs(os.path.join(args.save_path, 'pred_points'))

for i, (data, target) in enumerate(train_loader):
    data, target = data.cuda(args.gpu_id), target
    output = network(data)
    plot_xyz(target, save_path ='{0}/{1}/{2}.png'.format(args.save_path, 'target_points' ,i), xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    plot_xyz(output.detach().cpu(), save_path ='{0}/{1}/{2}.png'.format(args.save_path, 'pred_points' ,i), xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    if i%10 == 0:
        print(i)
