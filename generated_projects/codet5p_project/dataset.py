# Implement this for a project that: 

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
from numpy.random import RandomState
from sklearn.datasets import make_classification

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def train(epoch, model, data, target, criterion, optimizer, args):
    model.train()
    optimizer.zero_grad()
    output = model(data)

    if args.loss == 'CE':
        loss = criterion(output, target.long())
    elif args.loss == 'Hinge':
        loss = criterion(output, target.float())

    if args.reg == 'L2':
        loss += args.alpha * model.l2_loss()
    if args.reg == 'L1':
        loss += args.alpha * model.l1_loss()

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(epoch, model, data, target, criterion, optimizer, args):
    model.eval()
    output = model(data)

    if args.loss == 'CE':
        loss = criterion(output, target.long())
    elif args.loss == 'Hinge':
        loss = criterion(output, target.float())

    if args.reg == 'L2':
        loss += args.alpha * model.l2_loss()
    if args.reg == 'L1':
        loss += args.alpha * model.l1_loss()

    return loss.item()


def test(model, data, target, args):
    model.eval()
    output = model(data)

    if args.loss == 'CE':
        loss = F.cross_entropy(output, target.long(), size_average=False)
    elif args.loss == 'Hinge':