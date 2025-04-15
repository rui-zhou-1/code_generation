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
1. 模型类
   - 用于构建决策树
   - 可以支持多层决策树
   - 可以支持每个节点的权重参数（用于训练）
   - 支持多种分类方式（多分类、多分类+回归）
2. 数据加载类
   - 用于数据加载
   - 支持多种数据加载方式（csv、npy、h5等）
   - 支持每个数据加载格式化（数据处理）
   - 支持数据增强（数据增强）
   - 支持数据增强操作（数据增强）
   - 支持数据切割（数据切割）
3. 数据预处理类
   - 用于数据预处理
   - 支持多种数据预处理方式（数据处理）
   - 支持数据增强（数据增强）
   - 支持数据增强操作（数据增强）
   - 支持数据切割（数据切割）
4. 模型保存类
   - 用于模型保存
   - 支持多种模型保存方式（模型保存）
5. �

## Files
- model.py: Model architecture
- train.py: Training pipeline
- dataset.py: Data processing
- requirements.txt: Dependencies