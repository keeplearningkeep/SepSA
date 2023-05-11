import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

# 定义预处理操作
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),  # 随机裁剪为32x32，填充4个像素
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化
])

# 加载数据集并应用预处理操作
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


# 自定义标签 One-Hot 编码函数
def one_hot_label(label):
    num_classes = 10  # CIFAR-10 数据集中一共有 10 个类别
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1.0
    return one_hot


# 将标签进行 One-Hot 编码
train_set.targets = [one_hot_label(label) for label in train_set.targets]
test_set.targets = [one_hot_label(label) for label in test_set.targets]
