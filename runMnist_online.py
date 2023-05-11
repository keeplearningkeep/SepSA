import torch
import random
import torch.optim as optim
from torch.utils.data import DataLoader
"""# SepSA and Adam, RMS, SGD, NAG import this."""
import MyCNN_mnist_online as Net
"""# slimTrain import this"""
# import Slim_mnist_Kronecker_online as Net

import argparse

parser = argparse.ArgumentParser(description="RLS_Adam")
parser.add_argument("--seed", type=int, default=233, help="random seed (default: 233)")
parser.add_argument("--device", type=int, default=0, help="gpu device (default:0)")
parser.add_argument("--batch_size", type=int, default=1, metavar="batch_size", help="batch_size (default:64)")
parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--memory_depth", type=int, default=0, help="memory_depth")
args = parser.parse_args()

# 从文件中读取数据集
import pickle

with open('Datasets/mnist.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
train_set = loaded_data['train']
test_set = loaded_data['test']

result = {}
train = []
test = []
t = []
# run10 [699, 822, 644, 179, 538, 237, 820, 871, 496, 565]
# for seed in [699, 822, 644, 179, 538, 237, 820, 871, 496, 565]:
# for lr in [1e-3]:
for lr in [1e-3]:
    seed = 233
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    batch_size = args.batch_size

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    """The calculation results of training error and test error are the same at any batch size."""
    train_loader2 = DataLoader(train_set, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    # train_loader = DataLoader(MNIST.train_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(MNIST.test_set, batch_size=batch_size, shuffle=False)

    # 定义设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    net = Net.MyCNN(device).to(device)
    """Pass memory_depth parameters to the slimTrain network initialization"""
    # net = Net.MyCNN(device, args.memory_depth).to(device)

    optimizer_SepSA = optim.Adam([{'params': net.cnn.parameters(), 'weight_decay': 1e-5},
                                  {'params': net.fnn.parameters(), 'weight_decay': 1e-5}],
                                 lr=lr)
    optimizer_slimTrain = optim.Adam([{'params': net.cnn.parameters(), 'weight_decay': 1e-5},
                                      {'params': net.fnn.parameters(), 'weight_decay': 1e-5}],
                                     lr=lr)
    optimizer_Adam = optim.Adam([{'params': net.parameters(), 'weight_decay': 1e-5}],
                                lr=lr)
    optimizer_SGD = optim.SGD([{'params': net.parameters(), 'weight_decay': 1e-5}],
                              lr=lr)
    optimizer_Nest = optim.SGD([{'params': net.parameters(), 'weight_decay': 1e-5}],
                               lr=lr, nesterov=True, momentum=0.9)
    optimizer_RMS = optim.RMSprop([{'params': net.parameters(), 'weight_decay': 1e-5}],
                                  lr=lr, momentum=0.9)

    rls_train, rls_test, total_time = net.sgd_train(device, train_loader, train_loader2, test_loader, optimizer_SepSA,
                                                    num_epo=args.epochs, separate=True)
    # rls_train, rls_test, total_time = net.sgd_train(device, train_loader, train_loader2, test_loader, optimizer_slimTrain, num_epo=args.epochs, separate=True)
    # rls_train, rls_test, total_time = net.sgd_train(device, train_loader, train_loader2, test_loader, optimizer_Adam, num_epo=args.epochs, separate=False)

    result.update({f'train{lr}': rls_train, f'test{lr}': rls_test})
    # train.append(rls_train)
    # test.append(rls_test)
    # t.append(total_time)

# result = {'train': train, 'test': test, 'time': t}
with open('Result/MNIST/online/slim.pickle', 'wb') as f:
    pickle.dump(result, f)
