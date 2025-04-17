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

# model.py
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# dataset.py
from torch.utils.data import Dataset
import torch

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

#