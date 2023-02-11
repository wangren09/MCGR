import numpy as np
import os
import torch


import curves
import attack.pgd as pgd
import attack.pgd2 as pgd2
from attack.att import *
from attack.autopgd_train import apgd_train,pgd_1
def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None,pgdtype='0',curveflag=False):
    loss_sum = 0.0
    correct = 0.0
    #print(pgd)
    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda()
        target = target.cuda()
        t = input.data.new(1).uniform_()
        if pgdtype=='inf':
            at = pgd.PGD()
            if curveflag:
                input = at.generate(model, input, target, None, 0,t=t)
            else:
                input=at.generate(model, input, target, None, 0)
            model.train()
        if pgdtype=='2':
            at = pgd2.PGD()
            if curveflag:
                input = at.generate(model, input, target, None, 0,t=t)
            else:
                input = at.generate(model, input, target, None, 0)
            model.train()
        if pgdtype=='1':
            model.eval()
            if curveflag:
                input += pgd_l1_topk(model, input, target, epsilon=12, alpha=0.05, num_iter=10, device="cuda:0",
                                     restarts=0, version=0,t=t)
            else:
                #input+=pgd_l1_topk(model,input,target, epsilon=12, alpha=0.05, num_iter = 10, device = "cuda:0", restarts = 0, version = 0)
                input, acc_tr, _, _ = apgd_train(model, input, target, norm='L1',eps=12, n_iter=10)
            model.train()
        if pgdtype=='msd':
            if curveflag:
                input += msd_v0(model, input, target, epsilon_l_inf=8 / 255, epsilon_l_2=1, epsilon_l_1=12,
                                alpha_l_inf=2 / 255, alpha_l_2=0.2, alpha_l_1=0.05, num_iter=10, device="cuda:0", t=t)
            else:
                input += msd_v0(model,input,target,epsilon_l_inf = 8/255, epsilon_l_2= 1, epsilon_l_1 = 12,
                alpha_l_inf = 2/255, alpha_l_2 = 0.2, alpha_l_1 = 0.05, num_iter = 10, device = "cuda:0")
            model.train()
        if curveflag:
            output = model(input, t=t)
        else:
            output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda()
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
