# Implement this for a project that: 生成用于决策树分类的代码


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 导入数据集
def load_data(file_path):
    
    # 加载数据
    data_arr = np.loadtxt(file_path)
    # 切分数据集
    train_arr = data_arr[:200, :-1]
    test_arr = data_arr[200:, :-1]
    # 获取标签
    train_lable = data_arr[:200, -1]
    test_lable = data_arr[200:, -1]
    # 返回数据集
    return train_arr, test_arr, train_lable, test_lable

# 函数实现
def build_model(train_arr, train_lable, test_arr, test_lable, lr=0.01, iter_time=200):
    
    # 训练数据
    train_data_arr = np.array(train_arr)
    # 测试数据
    test_data_arr = np.array(test_arr)
    # 定义网络
    n_samples, n_features = train_data_arr.shape
    # 权重
    weights = np.random.randn(n_features, 1)
    # 初始化梯度
    gradient = np.zeros(weights.shape)
    # 初始化迭代次数
    iter_num = 0
    # 开始迭代
    while iter_num < iter_time:
        # 存储每次迭代训练数据的误差
        train_loss = 0
        # 迭代训练数据
        for i in range(n_samples):
            # 计算输出
            predict = np.dot(train_data_arr[i], weights)
            # 计算误差
            loss = train_lable[i] - predict
            # 梯度下降
            gradient = gradient + np.dot(train_data_arr[i].T, loss)
            # 更新梯度
            weights = weights + lr * gradient
            # 计算训练数据的误差
            train_loss += loss ** 2
        # 计算测试数据的误差
        test_loss = 0
        for i in range(n_samples):
            predict = np.dot(test_data_arr[i], weights)
            loss = test_lable[i] - predict
            test_loss += loss ** 2
        # 计算误差
        train_loss /= n_samples
        test_loss /= n_samples
        # 输出
        print('%d: train loss: %f, test loss: %f' % (iter_num, train_loss, test_loss))
        # 迭代次数+1
        iter_num += 1
    # 返回权重
    return weights

# 函数实现
def lr_plot(weights, train_arr, train_lable, test_arr, test_lable, iter_time=200):
    
    # 训练数据
    train_data_arr = np.array(train_arr)
    # 测试数据
    test_data_arr = np.array(test_arr)
    # 定义网络
    n_samples, n_features = train_data_arr.shape
    # 学习率
    lrs = np.linspace(0.01, 0.1, num=iter_time)
    # 存储每次迭代训练数据的误差
    train_losses = np.zeros(iter_time)
    # 存储每次迭代测试数据的误差
    test_losses = np.zeros(iter_time)
    # 迭代次数
    iter_num = 0
    # 开始迭代
    for lr in lrs:
        # 训练数据
        train_data_arr = np.array(train_arr)
        # 测试数据
        test_data_arr = np.array(test_arr)
        # 初始化梯度
        gradient = np.zeros(weights.shape)
        # 初始化迭代次数
        iter_num = 0
        # 开始迭代
        while iter_num < iter_time:
            # 存储每次迭代训练数据的误差
            train_loss = 0
            # 迭代训练数据
            for i in range(n_samples):
                # 计算输出
                predict = np.dot(train_data_arr[i], weights)
                # 计算误差
                loss = train_lable[i] - predict
                # 梯度下降
                gradient = gradient + np.dot(train_data_arr[i].T, loss)
                # 更新梯度
                weights = weights + lr * gradient
                # 计算训练数据的误差
                train_loss += loss ** 2
            # 计算测试数据的误差
            test_loss = 0
            for i in range(n_samples):
                predict = np.dot(test_data_arr[i], weights)
                loss = test_lable[i] - predict
                test_loss += loss ** 2
            # 计算误差
            train_loss /= n_samples
            test_loss /= n_samples
            # 输出
            print('%d: train loss: %f, test loss: %f' % (iter_num, train_loss, test_loss))
            # 迭代次数+1
            iter_num += 1
        # 计算�