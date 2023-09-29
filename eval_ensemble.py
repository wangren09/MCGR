import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

import data
import models
import utils

parser = argparse.ArgumentParser(description='Ensemble evaluation')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT', required=True,
                    help='checkpoint to eval, pass all the models through this parameter')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
criterion = F.cross_entropy

model.cuda()

ensemble_size = 0
predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

for path in args.ckpt:
    print(path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])

    predictions, targets = utils.predictions(loaders['test'], model)
    acc = 100.0 * np.mean(np.argmax(predictions, axis=1) == targets)

    predictions_sum += predictions
    ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

    print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))
