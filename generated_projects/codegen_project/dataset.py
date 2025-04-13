# 严格遵循以下规范：
# 1. 类和方法命名符合PEP8
# 2. 在dataset.py中实现对应模块职责
# 3. 使用类型注解提高可读性
# 4. 预留关键方法的docstring
# 5. 确保与其它模块的接口兼容性

实现代码：
1. 导入需要的包
2. 加载数据
3. 数据预处理
4. 数据增强
5. 模型定义
6. 模型训练
7. 模型评估
8. 模型保存
"""
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# 定义数据加载器
# 生成训练集，验证集，测试集
def get_loaders(batch_size):
    train_ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size)

    valid_ds = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    valid_loader = DataLoader(valid_ds, batch_size=batch_size)

    return train_loader, valid_loader

# 定义数据标准化和数据增强
# 对每个样本做标准化处理，并且对输入特征进行增强
# 增强：每个样本只增强一次
def get_transforms(input_size):
    return Compose([
        ToTensor(),
        Lambda(lambda x: x.view(-1, input_size)),
        Lambda(lambda x: x.div(255))
    ])

# 定义模型类，包含网络结构，前向传播逻辑，反向传播逻辑
# 定义损失函数
# 定义优化器
# 定义训练过程
# 定义模型保存函数
# 定义模型评估函数
# 定义模型训