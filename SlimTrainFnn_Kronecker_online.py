import torch
import math
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import slimtik_functions.slimtik_solve_kronecker_structure as tiksolvevec


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)


class MyNetwork(nn.Module):
    def __init__(self, input_size, h1_width=50, out_width=1, memory_depth=5, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05, Lambda=None):
        super(MyNetwork, self).__init__()
        f1 = nn.Linear(input_size, h1_width)
        act1 = nn.ReLU()
        self.feature_extractor = nn.Sequential(f1, act1)
        self.final_layer = nn.Linear(h1_width, out_width)
        self.apply(init_weights)

        W = self.final_layer.weight.data
        b = self.final_layer.bias.data.view(-1, 1)
        self.W = torch.cat([W, b], dim=1)

        # slimtik parameters
        self.memory_depth = memory_depth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.opt_method = opt_method
        self.reduction = reduction
        self.sumLambda = sumLambda

        # store history
        self.M = []

        # regularization parameters
        if Lambda is None:
            self.Lambda = self.sumLambda
        else:
            self.Lambda = Lambda

        self.LambdaHist = []

        self.alpha = None
        self.alphaHist = []

        # iteration counter
        self.iter = 0

        self.train_loss = []
        self.test_loss = []

    def forward(self, x, y=None, use_RLS=False):
        feature = self.feature_extractor(x)
        if use_RLS:
            with torch.no_grad():
                ones_column = torch.ones((feature.shape[0], 1))
                ZT = torch.cat((feature, ones_column), dim=1)
                Z = ZT.T

                # get batch size
                n_calTk = x.shape[0]

                # store history
                if self.iter > self.memory_depth:
                    self.M = self.M[1:]  # remove oldest

                # add current batch to history
                self.M.append(Z)

                # solve for W and b
                self.solve(Z, y.T, n_calTk)

                # update regularization parameter and iteration
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist += [self.alpha]

                W = self.W
                self.final_layer.weight.data = W[:, :-1]
                self.final_layer.bias.data = W[:, -1]
        return self.final_layer(feature)

    def solve(self, Z, C, n_calTk, dtype=torch.float32):
        M = torch.cat(self.M, dim=1)

        beta = 1.0
        if self.reduction == 'mean':
            beta = 1 / math.sqrt(n_calTk)

        W, info = tiksolvevec.solve(beta * Z, beta * C, beta * M, self.W,
                                    self.sumLambda, Lambda=self.Lambda, dtype=dtype, opt_method=self.opt_method,
                                    lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W = W
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

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
        print('Total training time = ', total_end - total_start)
        return self.train_loss, self.test_loss, total_end - total_start

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
        return loss

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
