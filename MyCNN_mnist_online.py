import torch
import time
import random
import math
import torch.nn as nn


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)


class MyCNN(nn.Module):
    def __init__(self, device):
        super(MyCNN, self).__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        bn1 = nn.BatchNorm2d(32)
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=2)

        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        bn2 = nn.BatchNorm2d(64)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d(kernel_size=2)

        fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        relu3 = nn.ReLU()
        self.cnn = nn.Sequential(conv1, bn1, relu1, pool1,
                                 conv2, bn2, relu2, pool2)
        self.fnn = nn.Sequential(fc1, relu3)
        self.final_layer = nn.Linear(in_features=128, out_features=10)
        self.apply(init_weights)
        self.device = device

        W = self.final_layer.weight.data.to(device)
        b = self.final_layer.bias.data.view(1, -1).to(device)
        self.W = torch.cat([torch.transpose(W, 0, 1), b], dim=0).to(device)
        self.P = 1 * torch.eye(self.W.shape[0]).to(device)
        
        # one epoch
        self.train_acc = []
        self.test_acc = []

    def forward(self, x, y=None, separate=False):
        feature_cnn = self.cnn(x).view(-1, 64 * 7 * 7)
        feature = self.fnn(feature_cnn)
        if separate:
            with torch.no_grad():
                ones_column = torch.ones((feature.shape[0], 1)).to(self.device)
                Phi = torch.cat((feature, ones_column), dim=1).to(self.device)
                self.W, self.P = RLS(Phi, y.to(self.device),
                                     self.W.to(self.device), self.P.to(self.device))
                W = torch.transpose(self.W, 0, 1).to(self.device)
                self.final_layer.weight.data = W[:, :-1]
                self.final_layer.bias.data = W[:, -1]
        return self.final_layer(feature)
        # return Phi @ self.W

    def sgd_train(self, device, train_loader, train_loader2, test_loader, optimizer, num_epo, separate):
        epochs = num_epo
        criterion = nn.MSELoss()
        # train_loss, train_acc = self.eval_loss_acc(train_loader2, criterion, device)
        test_loss, test_acc = self.eval_loss_acc(test_loader, criterion, device)
        # self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)
        print(f"iter {0}: test_loss:{test_loss:.4f} || test_acc: {test_acc:.4f}")
        
        total_start = time.time()
        for i in range(epochs):
            self.train_one_epoch(train_loader, train_loader2, test_loader, optimizer, criterion, device, separate)
        total_end = time.time()
        # train_loss, train_acc = self.eval_loss_acc(train_loader2, criterion, device)
        # test_loss, test_acc = self.eval_loss_acc(test_loader, criterion, device)
        # print(f"Epoch {i + 1}: train_loss: {train_loss:.4f}, test_loss:{test_loss:.4f} || "
        #           f"train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
        if separate:
            print('RLS Total training time = ', total_end - total_start)
        else:
            print('Adam Total training time = ', total_end - total_start)
            
        return self.train_acc, self.test_acc, total_end - total_start
        # return train_acc, test_acc, total_end - total_start

    def train_one_epoch(self, train_loader, train_loader2, test_loader, optimizer, criterion, device, separate):
        # self.train()
        # criterion.reduction = 'mean'
        k = 0
        for batch in train_loader:
            # online learning
            self.train()
            criterion.reduction = 'mean'
            data, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            y_pred = self.forward(data, target, separate)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            # one epoch
            k += 1
            if k % 500 == 0:
                # train_loss, train_acc = self.eval_loss_acc(train_loader2, criterion, device)
                test_loss, test_acc = self.eval_loss_acc(test_loader, criterion, device)
                # self.train_acc.append(train_acc)
                self.test_acc.append(test_acc)
                print(f"iter {k}: test_loss:{test_loss:.4f} || test_acc: {test_acc:.4f}")

    def eval_loss_acc(self, test_loader, criterion, device):
        self.eval()
        criterion.reduction = 'sum'
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                data, target = batch[0].to(device), batch[1].to(device)
                output = self.forward(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                for py, y in zip(output, target):
                    if torch.argmax(py) == torch.argmax(y):
                        correct += 1
                total += target.size(0)

        acc = 100. * correct / total
        avg_loss = test_loss / total

        return avg_loss, acc


def RLS(Phi, y, W, P):
    # Phi是一个mini_batch的特征
    # lam = 1
    for i in range(Phi.shape[0]):
        # 获取第i个样本的特征和标签
        xi = Phi[i, :].view(1, -1)
        yi = y[i, :].view(1, -1)
        Px = P @ xi.T
        # 更新参数W
        W = W + (Px @ (yi - xi @ W)) / (1 + xi @ Px)
        # 更新参数P
        P = P - Px @ (xi @ P) / (1 + xi @ Px)
        # 更新参数W
        # W = W + (P @ xi.T @ (yi - xi @ W)) / (lam + xi @ P @ xi.T)
        # # 更新参数P
        # P = (P - P @ xi.T @ xi @ P / (lam + xi @ P @ xi.T))/lam 

    return W, P
