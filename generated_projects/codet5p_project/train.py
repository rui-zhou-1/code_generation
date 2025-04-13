import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(Model, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def model(num_classes=1000):
    return Model(model=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_wsl'))

def model_cifar10(num_classes=10):
    return Model(model=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_wsl'))

def model_imagenet(num_classes=1000):
    return Model(model=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_wsl'))

def model_imagenet_cifar10(num_classes=10):
    return Model(model=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_wsl'))

def model_imagenet_cifar100(num_classes=100):
    return Model(model=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_wsl'))

def model_imagenet_svhn(num_classes=10):
    return Model(

from model import Model
from dataset import CustomDataset
