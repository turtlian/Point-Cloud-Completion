import dataset
from torch.utils.data import DataLoader
from model import TopNet
from metric import chamfer_distance
import os
import torch
import argparse
import torch.optim as optim
from utils import Logger, AverageMeter, draw_curve

parser = argparse.ArgumentParser(description='Point Cloud Completion Project')
parser.add_argument('--save_path', default='./exp2', type=str,
                    help='datapath')
parser.add_argument('--data_path', default='./shapenet', type=str,
                    help='datapath')
parser.add_argument('--model', default='topnet', type=str,
                    help='topnet or pcn')
parser.add_argument('--embedding_dim', default=1024, type=int,
                    help='embedding size')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adam', type=str,
                    help='optimizer')
parser.add_argument('--lr', default=0.5e-2, type=float,
                    help='learning rate')
parser.add_argument('--epochs', default=50, type=int,
                    help='train epoch')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight_decay')
parser.add_argument('--scaling', default=None, type=float,
                    help='scaling size')
parser.add_argument('--rotation', default=True, type=bool,
                    help='If "true", randomly rotate the point')
parser.add_argument('--mirror_prob', default=None, type=float,
                    help='Probability of randomly mirroring points')
parser.add_argument('--gpu_id', default=7, type=int,
                    help='gpu')
args = parser.parse_args()

def train(model, trn_loader, device, criterion, optimizer, epoch, num_epoch, train_logger):
    model.cuda(device)
    criterion.cuda(device)
    model.train()
    train_loss = AverageMeter()
    for i, (data, target) in enumerate(trn_loader):
        data, target = data.cuda(device), target.cuda(device)
        output = model(data)
        loss, _ = criterion(output.transpose(1,2), target.transpose(1,2))
        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(trn_loader), loss=loss))
    train_logger.write([epoch, train_loss.avg])

def test(model, tst_loader, device, criterion, epoch, num_epoch, val_logger):
    model.cuda(device)
    criterion.cuda(device)
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, (data, target) in enumerate(tst_loader):
            data, target = data.cuda(device), target.cuda(device)
            output = model(data)
            loss, _ = criterion(output.transpose(1,2), target.transpose(1,2))
            val_loss.update(loss.item())

        print("=================== TEST(Validation) Start ====================")
        print('Epoch : [{0}/{1}]  Test Loss : {loss:.4f}'.format(
                epoch, num_epoch, loss=val_loss.avg))
        print("=================== TEST(Validation) End ====================")

        val_logger.write([epoch, val_loss.avg])

def main():
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # define architecture
    if args.model == 'topnet':
        network = TopNet(encoder_feature_dim = args.embedding_dim, device = args.gpu_id)
    else:
        network = None
        print("pcn")
        # network = PCN()

    # load dataset
    train_dataset = dataset.ShapeNetDataset(args.data_path, mode='train', scaling = args.scaling,
                                            rotation = args.rotation, mirror_prob = args.mirror_prob)
    val_dataset = dataset.ShapeNetDataset(args.data_path, mode='val')
    print("Data Loaded")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # define criterion
    criterion = chamfer_distance()
    if args.optim == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logger
    train_logger = Logger(os.path.join(save_path, 'train_loss.log'))
    val_logger = Logger(os.path.join(save_path, 'val_loss.log'))

    # training & validation
    for epoch in range(1, args.epochs+1):
        train(network, train_loader, args.gpu_id, criterion ,optimizer, epoch, args.epochs, train_logger)
        test(network, val_loader, args.gpu_id ,criterion, epoch, args.epochs, val_logger)
        torch.save(network.state_dict(), '{0}/{1}_{2}.pth'.format(save_path, args.model ,epoch))
    draw_curve(save_path, train_logger, val_logger)
    print("Process complete")

if __name__ == '__main__':
    main()
