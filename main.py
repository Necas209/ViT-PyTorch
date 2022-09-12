import argparse
import os
import shutil

import timm
import torch
from tensorboard_logger import configure
from timm.models import VisionTransformer
from torch import optim, nn
from torch.backends import cudnn
from torchsummary import summary

from train import train, validate, test
from utils import load_data

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
parser.add_argument('--name', default='ViT-B-P8-224', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--model', default='vit_base_patch8_224', type=str,
                    help='name of ViT model to use')
parser.add_argument('--data',
                    help='Load data from dictionary (default: dataloader_dict.pt', action='store_true')

best_loss = None
args = argparse.Namespace


def main() -> None:
    global args, best_loss
    args = parser.parse_args()
    if args.tensorboard:
        configure(f"runs/{args.name}")
    # Load data
    train_loader, val_loader, test_loader = load_data(args)
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
    test(model, val_loader)


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
