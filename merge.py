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

    parser.add_argument('--ckpt', type=str, default='./realsave/pgd-pgd2-50_msd/checkpoint-50.pt', metavar='CKPT',
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
    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    pgdtest=pgdtest.PGDTest()
    model.cuda()
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state'])
    t = torch.FloatTensor([0.0]).cuda()
    t.data.fill_(0.82)
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
        '../test',
        50,
        model_state=model1.state_dict()
    )
    print('done!')
    columns = ['t', 'Test clean acc', 'Test pgd_inf acc', 'Test pgd_2 acc', 'Test clean loss', 'Test pgd_inf loss',
               'Test pgd_2 loss']
    teca, teia, te2a, tecl, teil, te2l, trca, tria, tr2a, trcl, tril, tr2l = pgdtest.test(model1, 0)
    values = [t, teca, teia, te2a, trcl, tril, tr2l]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    print(table)
import d
D=d.Data()
data_loader_test = torch.utils.data.DataLoader(dataset=D.data_test,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_correct = 0
for data in data_loader_test:
    X_test, y_test = data
    X_test, y_test = X_test.to(device), y_test.to(device)
    outputs = model1(X_test)
    _, pred = torch.max(outputs.data, 1)
    test_correct += torch.sum(pred == y_test.data)
print("Test Accuracy is:{:.4f}%".format(100 * test_correct / len(D.data_test)))





