import torch
import argparse
import numpy as np
import os
import tabulate
import pgdtest
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
                        help='training directory (default: /tmp/eval)')

    parser.add_argument('--num_points', type=int, default=61, metavar='N',
                        help='number of points on the curve (default: 61)')

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
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 12)')
    parser.add_argument('--gpus', type=str, default='',
                        help='GPUS')
    parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                        help='model name (default: None)')
    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')

    parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                        help='checkpoint to eval (default: None)')

    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

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
    if args.dataset == 'ImageNet100' and 'PreResNet' in args.model:
        architecture.kwargs['imgdim']=49
    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    pgdtest=pgdtest.PGDTest(args.dataset,args.batch_size,args.data_path)
    checkpoint = torch.load(args.ckpt)
    '''
    model.load_state_dict(checkpoint['model_state'])

    '''

    weights = checkpoint['model_state']

    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module.' in k  else k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict)
    model.cuda()
    if len(args.gpus)>0:
        gpus=list(map(int,args.gpus.split(',')))
        model=torch.nn.DataParallel(model,device_ids=gpus)
    T = args.num_points
    columns = ['t','Test clean acc','Test pgd_inf acc','Test pgd_2 acc','Test pgd1_acc','Train clean loss','Train pgd_inf loss','Train pgd_2 loss','Train pgd_1 loss']
    ts = np.linspace(0.0, 1.0, T)

    teca=np.zeros(T)
    teia = np.zeros(T)
    te2a = np.zeros(T)
    te1a= np.zeros(T)

    tecl = np.zeros(T)
    teil = np.zeros(T)
    te2l = np.zeros(T)
    te1l= np.zeros(T)

    trca = np.zeros(T)
    tria = np.zeros(T)
    tr2a = np.zeros(T)
    tr1a= np.zeros(T)

    trcl = np.zeros(T)
    tril = np.zeros(T)
    tr2l = np.zeros(T)
    tr1l= np.zeros(T)

    t = torch.FloatTensor([0.0]).cuda()
    f = open(args.dir+'/log.csv', 'w')
    print(args.dir+'/log.csv')
    for i, t_value in enumerate(ts):
        t.data.fill_(t_value)
        utils.update_bn(loaders['train'], model, t=t)
        teca[i],teia[i],te2a[i],tecl[i],teil[i],te2l[i],trca[i],tria[i],tr2a[i],trcl[i],tril[i],tr2l[i],te1a[i],te1l[i],tr1a[i],tr1l[i]=pgdtest.test(model,0,t=t)
        values = [t,teca[i],teia[i],te2a[i],te1a[i],trcl[i],tril[i],tr2l[i],tr1l[i]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    with f:
        write = csv.writer(f)
        write.writerow(['t','test clean acc','test pgd_inf acc','test pgd_2 acc','test pgd_1 acc','test clean loss','test pgd_inf loss','test pgd_2 loss','test pgd_1 loss',\
                            'train clean acc','train pgd_inf acc','train pgd_2 acc','train pgd_1 acc','train clean loss','train pgd_inf loss','train pgd_2 loss','train pgd_1 loss'])
        for i in range(T):
            values=[ts[i],teca[i],teia[i],te2a[i],te1a[i],tecl[i],teil[i],te2l[i],te1l[i],trca[i],tria[i],tr2a[i],tr1a[i],trcl[i],tril[i],tr2l[i],tr1l[i]]
            write.writerow(values)




