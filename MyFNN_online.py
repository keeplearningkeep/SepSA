import math
import random
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)


class MyNetwork(nn.Module):
    def __init__(self, input_size, h1_width=50, out_width=1):
        super(MyNetwork, self).__init__()
        f1 = nn.Linear(input_size, h1_width)
        act1 = nn.ReLU()
        self.feature_extractor = nn.Sequential(f1, act1)
        self.final_layer = nn.Linear(h1_width, out_width)
        self.apply(init_weights)

        W = self.final_layer.weight.data
        b = self.final_layer.bias.data.view(1, -1)
        self.W = torch.cat([torch.transpose(W, 0, 1), b], dim=0)
        self.P = 1 * torch.eye(self.W.shape[0])

        self.train_loss = []
        self.test_loss = []

    def forward(self, x, y=None, separate=False):
        feature = self.feature_extractor(x)
        if separate:
            with torch.no_grad():
                ones_column = torch.ones((feature.shape[0], 1))
                Phi = torch.cat((feature, ones_column), dim=1)
                self.W, self.P = RLS(Phi, y, self.W, self.P)
                W = torch.transpose(self.W, 0, 1)
                self.final_layer.weight.data = W[:, :-1]
                self.final_layer.bias.data = W[:, -1]
        return self.final_layer(feature)

    def sgd_train(self, train_data, test_data, optimizer, separate):
        epochs = 1
        criterion = nn.MSELoss()
        criterion.reduction = 'mean'
        init_loss = 0
        test_loss = self.eval_loss(criterion, test_data)
        self.train_loss.append(init_loss)
        self.test_loss.append(test_loss)
        print(f"iter {0}: train_loss: {init_loss:.4f}, test_loss:{test_loss:.4f}")
        total_start = time.time()
        for i in range(epochs):
            self.train_one_epoch(train_data, test_data, optimizer, criterion, separate)
        total_end = time.time()
        total_time = total_end - total_start
        print('Total training time = ', total_time)
        return self.train_loss, self.test_loss, total_time

    def train_one_epoch(self, train_data, test_data, optimizer, criterion, separate):
        k = 0
        run_data = []
        dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
        for batch in dataloader:
            self.train()
            optimizer.zero_grad()
            y_pred = self.forward(batch[0], batch[1], separate)
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()
            k += 1
            run_data.append(batch)
            train_loss = self.eval_runloss(criterion, run_data)
            test_loss = self.eval_loss(criterion, test_data)
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            print(f"iter {k}: train_loss: {train_loss:.4f}, test_loss:{test_loss:.4f}")

    def eval_loss(self, criterion, data):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(data[:][0])
            loss = criterion(y_pred, data[:][1])
        return loss.item()

    def eval_runloss(self, criterion, run_data):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            total_num = 0
            for data in run_data:
                y_pred = self.forward(data[0])
                loss = criterion(y_pred, data[1])
                total_loss += loss.item() * data[0].shape[0]
                total_num += data[0].shape[0]
        return total_loss / total_num


def RLS(Phi, y, W, P):
    # Phi是一个mini_batch的特征
    for i in range(Phi.shape[0]):
        # Get the features and labels of the i-th sample
        xi = Phi[i, :].view(1, -1)
        yi = y[i, :].view(1, -1)
        # update W
        px = P @ xi.T
        W = W - (px @ (xi @ W - yi)) / (1 + xi @ px)
        # update P
        P = P - px @ (xi @ P) / (1 + xi @ px)

        # # update W
        # W = W - (P @ xi.T @ (xi @ W - yi)) / (1 + xi @ P @ xi.T)
        # # update P
        # P = P - P @ xi.T @ xi @ P / (1 + xi @ P @ xi.T)

    return W, P
