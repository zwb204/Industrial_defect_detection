# coding: utf-8

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import time
import argparse
import numpy as np

import models
from dataset import ImageListDataset
from train import AverageMeter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cls_list = ['0_scratch',
            '1_gline',
            '2_bubble',
            '3_defect',
            '4_unformed',
            '5_foreign_matter',
            '6_burr',
            '7_lr',
            '8_pin']

def parse():
    args = argparse.ArgumentParser('model eval')
    args.add_argument('dataroot', type=str,
                    help='testset root dir')
    args.add_argument('testlist', type=str,
                    help='testset list file')
    args.add_argument('checkpoint', type=str,
                    help='checkpoint path')
    args.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    args.add_argument('--batch_size', type=int, default=64,
                    help='batch_size, default=64')
    args.add_argument('--num_workers', type=int, default=4,
                    help='DataLoader readers. default=4')

    return args.parse_args()


def cal_acc(dataloader, model, num_classes, device):
    accuracy = AverageMeter()
    model.eval()

    cls_count = np.zeros(num_classes, dtype=np.float32)
    cls_correct = np.zeros(num_classes, dtype=np.float32)

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            for gt_label in labels:
                cls_count[int(gt_label.item())] += 1

            _, preds = torch.max(outputs, 1)
            for corr_pred in labels[preds == labels.data]:
                cls_correct[int(corr_pred.item())] += 1

            acc = torch.mean((preds == labels.data).float())
            cls_acc = cls_correct / (cls_count + 1e-8)

            accuracy.update(acc.item(), batch_size)

            print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
            print('Test: [{0}/{1}] '
                  'Acc: {acc.val:.3f}({acc.avg:.3f})'
                  .format(i + 1, len(dataloader),
                          acc=accuracy), flush=True)

    return accuracy.avg, cls_acc


def eval(data_root, data_list, checkpoint, batch_size, num_workers):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_dataset = ImageListDataset(data_root, data_list, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    # model
    model = models.__dict__[args.arch](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    model = model.to(device)

    state = torch.load(checkpoint, map_location=device)
    state_dict = dict()
    for k, v in state['model'].items():
        state_dict[k.replace('module.','')] = v
    model.load_state_dict(state_dict)

    acc, cls_acc = cal_acc(test_loader, model, len(cls_list), device)

    print('=> Test Acc: %.4f\n' % acc)
    for i in range(len(cls_list)):
        print('%s acc: %.4f' % (cls_list[i], cls_acc[i]))


if __name__ == '__main__':
    args = parse()
    print('Eval...')
    eval(args.dataroot, args.testlist, args.checkpoint, args.batch_size, args.num_workers)
