import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
class Data:
    def __init__(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([transforms.ToTensor()])
        self.data_train = datasets.CIFAR10(root='./data/', train=True,
                                         download=True, transform=transform_train)
        self.data_test = datasets.CIFAR10(root='./data/', train=False,
                                        download=True, transform=transform)

if __name__ == '__main__':
   # D1=Data('mnist')
    D2=Data()
