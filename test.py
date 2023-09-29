import argparse
import torch
import curves
import d
import models
import attack.pgd as pgd
import attack.pgd2 as pgd2
from tqdm import tqdm
from attack.autopgd_train import apgd_train,pgd_1
from attack.att import  msd_v1,msd_v0,l1_dir_topk,pgd_l1_topk,pgd_worst_dir


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--transform', type=str, default='ResNet', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='PreResNet110', metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--ckpt', default='',help='ckpt')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

D=d.Data()

data_loader_test = torch.utils.data.DataLoader(dataset=D.data_test,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

files=[args.ckpt]
with open('result.txt', 'w') as f:
    out=""
    for file in files:
        out=out+file+'\n'
        architecture = getattr(models, args.model)

        if args.curve is None:
            model = architecture.base(num_classes=10, **architecture.kwargs)
        else:
            curve = getattr(curves, args.curve)
            model = curves.CurveNet(
                10,
                curve,
                architecture.curve,
                args.num_bends,
                args.fix_start,
                args.fix_end,
                architecture_kwargs=architecture.kwargs,
            )
            base_model = None
            if args.resume is None:
                for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                    if path is not None:
                        if base_model is None:
                            base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                        checkpoint = torch.load(path)
                        print('Loading %s as point #%d' % (path, k))
                        base_model.load_state_dict(checkpoint['model_state'])
                        model.import_base_parameters(base_model, k)
                if args.init_linear:
                    print('Linear initialization.')
                    model.init_linear()
        #model.load_state_dict(torch.load('./save/pgd/checkpoint-150.pt')['model_state'])
        model.load_state_dict(torch.load(file)['model_state'])
        model.cuda()
        model.eval()

        test_correct = 0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            test_correct += torch.sum(pred == y_test.data)
        print("Test Accuracy is:{:.4f}%".format(100 * test_correct / len(D.data_test)))
        out+="Test Accuracy is:{:.4f}%".format(100 * test_correct / len(D.data_test))+'\n'
        at=pgd.PGD()

        at2=pgd2.PGD()

        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test=at.generate(model,X_test,y_test,None,device)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after ATinf is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after ATinf is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'
        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test=at2.generate(model,X_test,y_test,None,device)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after AT2 is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after AT2 is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'


        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test += pgd_l1_topk(model,X_test,y_test, epsilon=12, alpha=0.05, num_iter = 50, device = "cuda:0", restarts = 0, version = 0)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after AT1 is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after AT1 is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'

        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test, acc_tr, _, _ = apgd_train(model, X_test, y_test, norm='L1',eps=12, n_iter=50)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after AT1-auto is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after AT1-auto is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'



        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test, acc_tr, _, _ = apgd_train(model, X_test, y_test, norm='L2',eps=1.0, n_iter=50)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after AT2-auto is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after AT2-auto is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'


        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test, acc_tr, _, _ = apgd_train(model, X_test, y_test, norm='Linf',eps=8/255, n_iter=50)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after ATinf-auto is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after ATinf-auto is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'



        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test += msd_v0(model, X_test,y_test, epsilon_l_inf = 8/255, epsilon_l_2= 1.0, epsilon_l_1 = 12,
                        alpha_l_inf = 2/255, alpha_l_2 = 0.2, alpha_l_1 = 0.05, num_iter = 50, device = "cuda:0")
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after AT-MSD is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after AT-MSD is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'

        correct=0
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_test += pgd_worst_dir(model, X_test,y_test, epsilon_l_inf = 8/255, epsilon_l_2= 1.0, epsilon_l_1 = 12,alpha_l_inf = 2/255, alpha_l_2 = 0.2, alpha_l_1 = 0.05, num_iter = 50, device = "cuda:0")
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
        print("Test Accuracy after AT-WORST is:{:.4f}%".format(100 * correct / len(D.data_test)))
        out+="Test Accuracy after AT-WORST is:{:.4f}%".format(100 * correct / len(D.data_test))+'\n'
    f.write(out)




