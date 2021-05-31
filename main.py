import dataset
from torch.utils.data import DataLoader
from model import TopNet, PCN
from loss import chamfer_distance
import os
import torch
import argparse
import torch.optim as optim
from utils import Logger, AverageMeter, draw_curve
import json
import torch.nn as nn
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='Point Cloud Completion Project')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='datapath')
parser.add_argument('--data_path', default='./shapenet', type=str,
                    help='datapath')
parser.add_argument('--npts', default=16384, type=int,
                    help='number of generated points')
parser.add_argument('--coarse', default=1024, type=int,
                    help='number of generated points')
parser.add_argument('--alpha', default=0.5, type=int,
                    help='coarse loss weight')
parser.add_argument('--model', default='topnet', type=str,
                    help='topnet or pcn')
parser.add_argument('--embedding_dim', default=1024, type=int,
                    help='embedding size')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adagrad', type=str,
                    help='optimizer')
parser.add_argument('--lr', default=0.1e-2, type=float,
                    help='learning rate')
parser.add_argument('--epochs', default=300, type=int,
                    help='train epoch')
parser.add_argument('--weight_decay', default=0.000001, type=float,
                    help='weight_decay')
parser.add_argument('--scaling', default=None, type=float,
                    help='scaling size')
parser.add_argument('--rotation', default=False, type=bool,
                    help='If "true", randomly rotate the point')
parser.add_argument('--mirror_prob', default=None, type=float,
                    help='Probability of randomly mirroring points')
parser.add_argument('--gpu_id', default='1', type=str, help='devices')
args = parser.parse_args()


def train(model, trn_loader, criterion, optimizer, epoch, num_epoch, train_logger):
    model.train()
    train_loss = AverageMeter()
    for i, (data, target, coarsegt, densegt) in enumerate(trn_loader):
        data, target, coarsegt, densegt = data.cuda(), target.cuda(), coarsegt.cuda(), densegt.cuda()
        output = model(data)
        if args.model == 'topnet':
            loss, _ = criterion(output.transpose(1,2), densegt.transpose(1,2))
        elif args.model == 'pcn':
            loss1, _ = criterion(output[1].transpose(1,2), coarsegt.transpose(1,2))
            loss2, _ = criterion(output[2].transpose(1, 2), densegt.transpose(1, 2))
            loss = args.alpha * loss1 + loss2
        train_loss.update(loss.item()*10000)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(trn_loader), loss=loss*10000))
    train_logger.write([epoch, train_loss.avg])


def test(model, tst_loader, criterion, epoch, num_epoch, val_logger):
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, (data, target, coarsegt, densegt) in enumerate(tst_loader):
            data, target, coarsegt, densegt = data.cuda(), target.cuda(), coarsegt.cuda(), densegt.cuda()
            output = model(data)
            if args.model == 'topnet':
                loss, _ = criterion(output.transpose(1, 2), densegt.transpose(1, 2))
            elif args.model == 'pcn':
                loss1, _ = criterion(output[1].transpose(1, 2), coarsegt.transpose(1, 2))
                loss2, _ = criterion(output[2].transpose(1, 2), densegt.transpose(1, 2))
                loss = args.alpha * loss1 + loss2
            val_loss.update(loss.item()*10000)

        print("=================== TEST(Validation) Start ====================")
        print('Epoch : [{0}/{1}]  Test Loss : {loss:.4f}'.format(
                epoch, num_epoch, loss=val_loss.avg))
        print("=================== TEST(Validation) End ======================")
        val_logger.write([epoch, val_loss.avg])


def main():
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # define architecture
    if args.model == 'topnet':
        network = TopNet(encoder_feature_dim = args.embedding_dim, decoder_feature_dim = 8, npts = args.npts).cuda()
    else:
        network = PCN(encoder_feature_dim = args.embedding_dim, num_coarse = args.coarse ,num_dense = args.npts).cuda()
    network = nn.DataParallel(network).cuda()

    # load dataset
    train_dataset = dataset.ShapeNetDataset(args.data_path, mode='train', point_class = 'all',
                                            scaling = args.scaling, rotation = args.rotation,
                                            mirror_prob = args.mirror_prob, num_coarse = args.coarse, num_dense = args.npts)
    val_dataset = dataset.ShapeNetDataset(args.data_path, mode='val', point_class = 'all')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Data Loaded {len(train_dataset)}")

    # define criterion
    criterion = chamfer_distance().cuda()
    if args.optim == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.7)

    # logger
    train_logger = Logger(os.path.join(save_path, 'train_loss.log'))
    val_logger = Logger(os.path.join(save_path, 'val_loss.log'))

    # training & validation
    for epoch in range(1, args.epochs+1):
        train(network, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
        test(network, val_loader, criterion, epoch, args.epochs, val_logger)
        scheduler.step()
        if epoch%20 == 0 or epoch == args.epochs :
            torch.save(network.state_dict(), '{0}/{1}_{2}.pth'.format(save_path, args.model ,epoch))
    draw_curve(save_path, train_logger, val_logger)
    print("Process complete")

if __name__ == '__main__':
    main()
