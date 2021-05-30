import dataset
from model import TopNet
import os
from utils import plot_xyz, plot_pcds
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from utils import AverageMeter
from loss import chamfer_distance

parser = argparse.ArgumentParser(description='visualization')
parser.add_argument('--model_path', default='./scale0_rot1_mi05_ada0001_plane/topnet_140.pth', type=str,
                    help='model path')
parser.add_argument('--p_class', default='car', type=str,
                    help='class')
parser.add_argument('--data_path', default='./shapenet', type=str,
                    help='data path')
parser.add_argument('--data', default='shapenet', type=str,
                    help='dataset name')
parser.add_argument('--gpu_id', default=4, type=int,
                    help='gpu')
parser.add_argument('--mode', default='test', type=str,
                    help='test/visual')
args = parser.parse_args()

# load setting
path = args.model_path.split('/')[1]
path = os.path.join(path, 'configuration.json')
with open(path, 'r') as f:
    configuration = json.load(f)
save_path = configuration['save_path']
npts = configuration['npts']
model = configuration['model']
embedding_dim = configuration['embedding_dim']

# load dataset
if args.data == 'shapenet':
    test_dataset = dataset.ShapeNetDataset(args.data_path, mode='val', point_class = args.p_class)
#    train_dataset = dataset.ShapeNetDataset(args.data_path, mode='train', point_class = 'plane')
elif args.data == 'kitti':
    test_dataset = dataset.KittiDataset(args.data_path)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

# load trained model & loss
if configuration['model'] == 'topnet':
    network = TopNet(embedding_dim, 8, npts)
elif configuration['model'] == 'pcn':
    network = None
network=network.cuda(args.gpu_id)
network.load_state_dict(torch.load(args.model_path))
network.eval()
criterion = chamfer_distance().cuda(args.gpu_id)

# visualization
def visualization():
    if not os.path.exists(os.path.join(save_path, 'target_points', args.p_class)):
        os.makedirs(os.path.join(save_path, 'input_points', args.p_class))
        os.makedirs(os.path.join(save_path, 'target_points', args.p_class))
        os.makedirs(os.path.join(save_path, 'pred_points', args.p_class))

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(args.gpu_id), target
        output = network(data)
        plot_xyz(data.detach().cpu(), save_path='{0}/{1}/{2}.png'.format(save_path, 'input_points', args.p_class, i),
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        plot_xyz(target, save_path ='{0}/{1}/{2}.png'.format(save_path, 'target_points', args.p_class, i),
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        plot_xyz(output.detach().cpu(), save_path ='{0}/{1}/{2}.png'.format(save_path, 'pred_points', args.p_class, i),
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        if i%10 == 0:
            print(i)

def test():
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(args.gpu_id), target.cuda(args.gpu_id)
            output = network(data)
            if args.model == 'topnet':
                loss, _ = criterion(output.transpose(1, 2), target.transpose(1, 2))
            elif args.model == 'pcn':
                loss, _ = criterion(output[2].transpose(1, 2), target.transpose(1, 2))
            val_loss.update(loss.item()*10000)

        print("=================== TEST(Validation) Start ====================")
        print('Class : {p_class}  Test Loss : {loss:.4f}'.format(
                p_class = args.p_class, loss=val_loss.avg))
        print("=================== TEST(Validation) End ======================")

if args.mode == 'visual':
    visualization()
elif args.mode == 'test':
    test()
