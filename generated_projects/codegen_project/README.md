# CodeGen Project

## Original Description
生成决策树分类代码

## Expanded Requirements
# Expand project description into detailed technical requirements with module specs
Original description: 生成决策树分类代码

请生成：
1. 详细技术需求（包含输入输出规格）
2. 模块划分规范：
   - model.py职责：神经网络结构定义、前向传播逻辑
   - dataset.py职责：数据加载、预处理、增强操作
   - train.py职责：训练循环、验证逻辑、模型保存
3. 模块间交互要求：
   - train.py需要调用model中的Model类和dataset中的DataLoader类
   - dataset的输出格式需与model的输入维度匹配

详细需求：

1. 模型结构：
   - 层：卷积层、池化层、卷积层、池化层、全连接层、输出层
   - 激活函数：relu、sigmoid、tanh、softmax
   - 损失函数：交叉熵
   - 梯度下降方法：
      - SGD：随机梯度下降
      - Adam：自适应梯度下降
      - RMSProp：RMSprop
      - AdaGrad：AdaGrad
      - AdaDelta：AdaDelta
      - Adamax：Adamax
      - Nadam：Nadam
      - AdaDelta：AdaDelta
2. 数据加载：
   - 网络结构：
      - 卷积层、池化层、卷积层、池化层、全连接层、输出层
      - 输入数据格式：NCHW
      - 输出数据格式：NHWC
      - 输入数据大小：[N,C,H,W]
      - 输出数据大小：[N,H,W,C]
      - 输入数据大小：[N,C,H,W]
      - 输出数据大小：[N,H,W,C]
      - 输入数据大小：[N,H,W,C]
      - 输出数据大小：[N,H,W,C]
      - 激活函数：relu、sigmoid、tanh
      - 损失函数：交叉熵
      - 梯度下降方法：
         - SGD：随机梯度下降

## Files
- model.py: Model architecture
- train.py: Training pipeline
- dataset.py: Data processing
- requirements.txt: Dependencies