from __future__ import annotations

import argparse
import time
from pprint import pprint

import pandas as pd
import torch.nn as nn
import torch.nn.parallel
from sklearn.metrics import mean_squared_error, r2_score
from tensorboard_logger import log_value
from timm.models.vision_transformer import VisionTransformer
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import AverageMeter


def train(model: VisionTransformer, train_loader: DataLoader, criterion: nn.MSELoss, optimizer: Adam, epoch: int,
          args: argparse.Namespace) -> None:
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.perf_counter()
    for i, (inp, target) in enumerate(train_loader):
        target: torch.Tensor = target.cuda(non_blocking=True)
        inp: torch.Tensor = inp.cuda()

        # compute output
        output: torch.Tensor = model(inp)
        loss: torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss, inp.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)


def validate(model: VisionTransformer, val_loader: DataLoader, criterion: nn.MSELoss, epoch: int,
             args: argparse.Namespace) -> float | torch.Tensor:
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.perf_counter()
    for i, (inp, target) in enumerate(val_loader):
        target: torch.Tensor = target.cuda(non_blocking=True)
        inp: torch.Tensor = inp.cuda()

        # compute output
        with torch.no_grad():
            output: torch.Tensor = model(inp)
            loss: torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss, inp.size(0))

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        print(f'Validation: [{i}/{len(val_loader)}]\t'
              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')

    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
    return losses.avg


def test(model: VisionTransformer, loader: DataLoader) -> None:
    """Perform testing on the test set"""
    # switch to evaluate mode
    model.eval()

    # actual and predicted y values
    t_y = torch.Tensor().cuda()
    t_yhat = torch.Tensor().cuda()

    for i, (inp, target) in enumerate(loader):
        target: torch.Tensor = target.cuda(non_blocking=True)
        inp: torch.Tensor = inp.cuda()

        # compute output
        with torch.no_grad():
            output: torch.Tensor = model(inp)

        # store y and yhat values
        t_y = torch.cat((t_y, target))
        t_yhat = torch.cat((t_yhat, output))

    # Calculate RMSE loss and R^2
    evaluate(t_y, t_yhat)


def evaluate(t_y: torch.Tensor, t_yhat: torch.Tensor) -> None:
    """Evaluates model in terms of RMSE loss and R^2"""
    # assure tensors are in CPU
    if t_y.is_cuda:
        t_y = t_y.cpu()
    if t_yhat.is_cuda:
        t_yhat = t_yhat.cpu()

    # convert tensors to numpy arrays
    y = t_y.numpy()
    yhat = t_yhat.numpy()

    # compute RMSE loss and R^2
    rmse_loss = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)

    # print actual and predicted values
    df = pd.DataFrame(data={'Actual': y.flatten(), 'Predicted': yhat.flatten()})
    pprint(df)
    print(f'RMSE loss = {rmse_loss}')
    print(f'R^2 = {r2}')
