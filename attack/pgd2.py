import torch
import torch.nn as nn


class PGD:
    def __init__(self, eps=1.0, step_size=0.2, max_iter=10, random_init=True,
                 targeted=False, loss_fn=nn.CrossEntropyLoss(), batch_size=64,eps_for_division=1e-10):
        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter
        self.random_init = random_init
        self.targeted = targeted
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.eps_for_division=eps_for_division

    def attack(self, model, x, y, x_adv=None, targets=None,**kwargs):
        batchsize=x.shape[0]
        if x_adv is None:
            if self.random_init:
                delta = torch.empty_like(x).normal_()
                d_flat = delta.view(x.size(0), -1)
                n = d_flat.norm(p=2, dim=1).view(x.size(0), 1, 1, 1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r / n * self.eps
                x_adv = torch.clamp(x + delta, min=0, max=1).detach()

            else:
                x_adv = torch.clone(x).detach()
        x_adv.requires_grad_(True)
        if 't' in kwargs:
            t = kwargs['t']
            t = float(t.item())
            pred_adv = model(**dict(input=x_adv,t=t))
        else:
            pred_adv = model(x_adv,**kwargs)
        if isinstance(pred_adv, (list, tuple)):
            pred_adv = pred_adv[-1]
        if self.targeted:
            assert targets is not None, "Target labels not found!"
            loss = self.loss_fn(pred_adv, targets)
        else:
            loss = self.loss_fn(pred_adv, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norms = torch.norm(grad.view(batchsize, -1), p=2, dim=1) + self.eps_for_division
        grad = grad / grad_norms.view(batchsize, 1, 1, 1)
        x_adv = x_adv.detach() + self.step_size * grad

        delta = x_adv - x
        delta_norms = torch.norm(delta.view(batchsize, -1), p=2, dim=1)
        factor = self.eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)

        x_adv = torch.clamp(x + delta, min=0, max=1).detach()

        return x_adv

    def generate(self, model, x, y=None, targets=None, device=torch.device("cpu"),**kwargs):
        model.to(device)
        model.eval()
        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i + self.batch_size].to(device)
            if y is None:
                if 't' in kwargs:
                    t = kwargs['t']
                    t = float(t.item())
                    y_batch = model(**dict(input=x_batch,t=t))
                else:
                    y_batch = model(x_batch,**kwargs)
                if isinstance(y_batch, tuple):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(device)
            else:
                y_batch = y[i: i + self.batch_size].to(device)
            for j in range(self.max_iter):
                if j == 0:
                    x_adv_batch = self.attack(model, x_batch, y_batch, targets=targets,**kwargs)
                else:
                    x_adv_batch = self.attack(model, x_batch, y_batch, x_adv_batch, targets=targets,**kwargs)
            x_adv.append(x_adv_batch)
        return torch.cat(x_adv, dim=0).to(device)

