import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import attack.pgd as pgd
import attack.pgd2 as pgd2
from attack.att import *
from tqdm import tqdm

class PGDTest():
    def __init__(self):
        self.data_train = datasets.CIFAR10(root='./data/cifar10', train=True,
                                     download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.data_loader_train = torch.utils.data.DataLoader(dataset=self.data_train,
                                                       batch_size=128,
                                                       shuffle=True,
                                                       num_workers=8)
        self.data_test = datasets.CIFAR10(root='./data/cifar10', train=False,
                                     download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.data_loader_test = torch.utils.data.DataLoader(dataset=self.data_test,
                                                       batch_size=128,
                                                       shuffle=True,
                                                       num_workers=8)
    def test_once(self,model,train,device,pgdtype,**kwargs):
        correct = 0
        cost = torch.nn.CrossEntropyLoss()
        loss=0.0
        data_loader_test=self.data_loader_test
        num=len(self.data_test)
        if train:
            data_loader_test = self.data_loader_train
            num = len(self.data_train)
        at = pgd.PGD()
        at2 = pgd2.PGD()
        for data in tqdm(data_loader_test):
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            if pgdtype == 'inf':
                X_test = at.generate(model, X_test, y_test, None, device,**kwargs)
            if pgdtype=='2':
                X_test = at2.generate(model, X_test, y_test, None, device,**kwargs)
            if pgdtype=='1':
                X_test+= pgd_l1_topk(model, X_test, y_test, epsilon=12, alpha=0.05, num_iter=50, device="cuda:0", restarts=0,
                               version=0,**kwargs)
            outputs = model(X_test, **kwargs)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y_test.data)
            l = cost(outputs, y_test)
            loss += l.item()
        return float(100 * correct / num),loss / (int(num/128) + 1)
    def test(self,model,device,**kwargs):
        model.to(device)
        model.eval()
        teca, tecl = self.test_once(model, False, device, '0', **kwargs)
        teia, teil = self.test_once(model, False, device, 'inf', **kwargs)
        te2a, te2l = self.test_once(model, False, device, '2', **kwargs)
        te1a, te1l = self.test_once(model, False, device, '1', **kwargs)
        
        trca, trcl = self.test_once(model, True, device, '0', **kwargs)
        tria, tril = self.test_once(model, True, device, 'inf', **kwargs)
        tr1a, tr1l = self.test_once(model, True, device, '1', **kwargs)
        tr2a, tr2l = self.test_once(model, True, device, '2', **kwargs)

        return teca,teia,te2a,tecl,teil,te2l,trca,tria,tr2a,trcl,tril,tr2l,te1a,te1l,tr1a,tr1l


'''
if __name__ == '__main__':
    model=models.preresnet.PreResNet110.base(10)
    model.load_state_dict(torch.load('./save/pgd/checkpoint-150.pt')['model_state'])
    model.to(0)
    pgdtest=PGDTest()
    pgdtest.test(model,0)
'''
