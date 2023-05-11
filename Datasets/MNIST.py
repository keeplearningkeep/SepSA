from torchvision.datasets import MNIST
import torchvision.transforms as t
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# 数据预处理
transform = t.Compose([
        t.ToTensor(),
        t.Normalize((0.1307,), (0.3081,))
    ])

# 获取数据集
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train = DataLoader(train_dataset, batch_size=60000, shuffle=False)
test = DataLoader(test_dataset, batch_size=10000, shuffle=False)
for i, (x, y) in enumerate(train):
    x_train = x
    y_train = F.one_hot(y, num_classes=10)
for i, (x, y) in enumerate(test):
    x_test = x
    y_test = F.one_hot(y, num_classes=10)


train_set = TensorDataset(x_train.float(), y_train.float())
test_set = TensorDataset(x_test.float(), y_test.float())




