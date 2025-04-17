# 实现要求：
# 1. 类和方法命名符合PEP8
# 2. 在model.py中实现对应模块职责
# 3. 使用类型注解提高可读性
# 4. 编写必要的docstring
# 5. 确保与其它模块的接口兼容性

请根据上述要求写出具体的代码实现，确保所有接口都被正确实现，不要仅包含注释。例如，对于model.py，必须包含Model类的实现，包括__init__和forward方法。

代码实现：
# model.py
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.linear(input)
        return output

# dataset.py
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_data(train_batch_size, test_batch_size):
    train_loader = DataLoader(
        datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=ToTensor(),
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            "data",
            train=False,
            download=True,
            transform=ToTensor(),
        ),
        batch_size=test_batch_size,
        shuffle=True,
    )
    return train_loader, test_loader

# train.py
from torch import optim

def train(model, train_loader, loss_fn, optimizer, device):
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, test_loader, loss_fn, device):
    size = len(test_loader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).