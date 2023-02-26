import torch
import argparse
import numpy as np
import os
import tabulate
import models
import curves
import utils
import data
import csv

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cost = torch.nn.CrossEntropyLoss()
    parser = argparse.ArgumentParser(description='DNN curve evaluation')
    parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                        help='save directory')

    parser.add_argument('--num_points', type=int, default=61, metavar='N',
                        help='number of points on the curve (default: 61)')

    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--transform', type=str, default='ResNet', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default='data', metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 12)')

    parser.add_argument('--model', type=str, default='PreResNet110', metavar='MODEL',
                        help='model name (default: None)')
    parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')

    parser.add_argument('--ckpt', type=str, default='./finalsave/0.3-0.8-50/checkpoint-50.pt', metavar='CKPT',
                        help='checkpoint to merge (default: None)')

    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--t', type=float, default=0.5, metavar='T',
                        help='t (default: 0.5)')

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        shuffle_train=False
    )

    architecture = getattr(models, args.model)
    curve = getattr(curves, args.curve)
    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    model.cuda()
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state'])
    t = torch.FloatTensor([0.0]).cuda()
    t.data.fill_(args.t)
    utils.update_bn(loaders['train'], model, t=t)
    parm = []
    parms=[]
    bezier=curves.Bezier(3).cuda()
    al=bezier(t)
    i=0
    model1 = models.preresnet.PreResNet110.base(10)
    m = model.state_dict()
    m1 = model1.state_dict()
    dict={}
    for p in m:
        #print(p)
        bool = True
        for i in range(3):
            tar='_'+str(i)
            if p[-2:] == tar:
                bool = False
                name = p[4:-2]
                #print(name)
                if dict.get(name) is None:
                    dict[name] = al[i] * m[p]
                else:
                    dict[name] += al[i] * m[p]
        if bool :
            name = p[4:]
            dict[name] = m[p]
    for p in m1:
        #print(p)
        if not dict.get(p) is None:
            #print(p)
            m1[p]=dict[p]
        else:
            m1[p]=dict[p[:-8]+'weight']
    model1.load_state_dict(m1)
    utils.save_checkpoint(
        args.dir,
        0,
        model_state=model1.state_dict()
    )
    print('done!')





