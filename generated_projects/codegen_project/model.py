# 实现要求：
# 1. 所有导入语句必须位于文件最顶部
# 2. 类和方法命名符合PEP8
# 3. 使用类型注解提高可读性
# 4. 编写必要的docstring
# 5. 确保与其它模块的接口兼容性
# 6. 示例：对于model.py，应包含：
#    from torch import nn
#    import torch
#    class Model(nn.Module):
# 请确保模块间的导入正确，例如train.py需要导入model.Model和dataset.Dataset

请根据上述要求写出具体的代码实现，确保所有接口都被正确实现，不要仅包含注释。例如，对于model.py，必须包含Model类的实现，包括__init__和forward方法，且所有导入语句在文件顶部。

代码实现：
# 导入所需模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义超参数
PATH = './model.pth'

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义数据加载器
def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(