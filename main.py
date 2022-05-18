import argparse
import os
import shutil

import numpy as np
import scipy.io as sio
import timm
from tensorboard_logger import configure
from torch import optim
from torch.backends import cudnn
from torch.utils import data
from torch.utils.data import TensorDataset
from torchsummary import summary

from train import *

parser = argparse.ArgumentParser(description='PyTorch ViT Training')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    help='path to latest checkpoint (default: --name)', action='store_true')
parser.add_argument('--best',
                    help='Load best model (default: --name)', action='store_true')
parser.add_argument('--name', default='ViT-B-P16-224', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--model', default='vit_base_patch16_224', type=str,
                    help='name of ViT model to use')
parser.add_argument('--data',
                    help='Load data from dictionary (default: dataloader_dict.pt', action='store_true')

best_loss = None
args = Namespace


def main():
    global args, best_loss
    args = parser.parse_args()
    if args.tensorboard:
        configure(f"runs/{args.name}")
    # Data loading
    if args.data is False:
        mat = sio.loadmat('Hyper_2012_TF_sugar.mat')
        x: np.ndarray = mat['MeanROffPedNorm']
        y: np.ndarray = mat['sugar']
        x = x.transpose()[:, 10:-6].reshape(240, 1, 32, 32)
        t_x = torch.Tensor(x)
        t_y = torch.Tensor(y)
        # Shuffle tensors' rows
        indices = torch.randperm(t_x.size()[0])
        t_x = t_x[indices]
        t_y = t_y[indices]
        # Split dataset into train, validation and test sets
        dataset = TensorDataset(t_x, t_y)
        ds_size = len(dataset)
        test_size = int(0.1 * ds_size)
        train_size = ds_size - test_size
        ds, test_ds = data.random_split(dataset, [train_size, test_size])
        ds_size = len(ds)
        val_size = int(0.1 * ds_size)
        train_size = ds_size - val_size
        train_ds, val_ds = data.random_split(ds, [train_size, val_size])
        # Create data loaders
        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        # Save data loaders to file
        if args.resume is False:
            d = {'train_loader': train_loader,
                 'val_loader': val_loader,
                 'test_loader': test_loader}
            torch.save(d, f'./runs/{args.name}/dataloader_dict.pt')
    else:
        dl_path = f'./runs/{args.name}/dataloader_dict.pt'
        print(f'Loading data from {dl_path}')
        d = torch.load(dl_path)
        train_loader = d['train_loader']
        val_loader = d['val_loader']
        test_loader = d['test_loader']
    # Model creation
    model = timm.create_model(args.model,
                              pretrained=True,
                              in_chans=1,
                              num_classes=1,
                              img_size=(32, 32)
                              )
    model: VisionTransformer = model.cuda()
    summary(model, input_size=(1, 32, 32))
    cudnn.benchmark = True
    # optionally resume from a checkpoint
    if args.resume:
        if args.best:
            args.epochs = args.start_epoch
            cp_path = f'./runs/{args.name}/model_best.pth.tar'
            print(f"=> loading best model '{args.name}'")
        else:
            cp_path = f'./runs/{args.name}/checkpoint.pth.tar'
        if os.path.isfile(cp_path):
            print(f"=> loading checkpoint '{cp_path}'")
            checkpoint = torch.load(cp_path)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{cp_path}' (epoch {checkpoint['epoch']})")
            # Restore data loaders
            d = torch.load(f'./runs/{args.name}/dataloader_dict.pt')
            train_loader = d['train_loader']
            val_loader = d['val_loader']
            test_loader = d['test_loader']
        else:
            print(f"=> no checkpoint found at '{cp_path}'")
    # Define loss function and optimizer for regression
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters())
    # Train and validate model
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch, args)
        # evaluate on validation set
        loss = validate(model, val_loader, criterion, epoch, args)
        # remember the best loss and save checkpoint
        if best_loss is None:
            best_loss = loss
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best)
    print('Best loss: ', best_loss)
    # Test and evaluate model
    print("--- Model testing ---")
    test(model, test_loader)
    print("--- Model validation ---")
    val_and_eval(model, val_loader)


def save_checkpoint(state: dict, is_best: bool, filename: str = 'checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = f"./runs/{args.name}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'runs/{args.name}/model_best.pth.tar')


if __name__ == '__main__':
    main()
