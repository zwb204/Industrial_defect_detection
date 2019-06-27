# coding: utf-8

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import time
import os
import shutil
import math

import models
from config import Args
from dataset import ImageListDataset


best_acc = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_learning_rate(optimizer, epoch, iter_size, iter_num, args):
    current_iter = epoch * iter_size + iter_num
    if current_iter < args.warm_up:
        current_lr = args.lr * math.pow(current_iter / args.warm_up, 4)
    else:
        current_lr = args.lr * (1 + math.cos(epoch * math.pi / args.epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def train(dataloader, model, criterion, optimizer, epoch, args):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    tic = time.time()
    for i, (images, labels) in enumerate(dataloader):
        lr = set_learning_rate(optimizer, epoch, len(dataloader), i, args)
        batch_size = images.size(0)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        outputs = model(images) # shape=(b, n_classes)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
            print('Train Epoch: [{0}][{1}/{2}] '
                  'Batch Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Loss {loss.val:.3f}({loss.avg:.3f}) '
                  'Lr {lr:.6f}'
                  .format(epoch, i, len(dataloader),
                          batch_time=batch_time,
                          loss=losses,
                          lr=lr), flush=True)


def val(dataloader, model, criterion, args):
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), batch_size)

            _, preds = torch.max(outputs, 1)
            acc = torch.mean((preds == labels.data).float())

            accuracy.update(acc.item(), batch_size)

            if i % args.print_freq == 0:
                print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
                print('Val: [{0}/{1}] '
                      'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                      'Acc: {acc.val:.3f}({acc.avg:.3f})'
                      .format(i, len(dataloader),
                              loss=losses,
                              acc=accuracy), flush=True)

    return losses.avg, accuracy.avg


def main(args):
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("Use GPU: {} for training.".format(args.gpus))

    # model
    model = models.__dict__[args.arch](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)

    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.checkpoint:
        print('=> loading checkpoint from {}...'.format(args.checkpoint))
        state = torch.load(args.checkpoint)
        args.start_epoch = state['epoch']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    # train dataset
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomRotation((-5, 5)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = ImageListDataset(args.data_root, args.train_list, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    if args.val_list:
        # val dataset
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            transforms.Normalize([0.5], [0.5])
        ])
        val_dataset = ImageListDataset(args.data_root, args.val_list, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epoch):
        global best_acc
        train(train_loader, model, criterion, optimizer, epoch, args)
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(args.checkpoint_dir,
                                       'checkpoint_epoch_{:04d}.pth.tar'.format(state['epoch']))
        torch.save(state, checkpoint_file)

        if args.val_list:
            print('val...')
            val_loss, val_acc = val(val_loader, model, criterion, args)
            print('Val Loss: {loss:.3f}, Val Acc: {acc:.3f}'.format(loss=val_loss, acc=val_acc))

            is_best = (val_acc > best_acc)
            if is_best:
                best_acc = val_acc
                best_checkpoint_file = os.path.join(args.checkpoint_dir,
                                                    'checkpoint_best.path.tar')
                shutil.copy2(checkpoint_file, best_checkpoint_file)


if __name__ == '__main__':
    main(Args)
