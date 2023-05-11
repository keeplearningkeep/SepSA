import torch
import math
import time
import torch.nn as nn
import slimtik_functions.slimtik_solve_kronecker_structure as tiksolvevec


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)


class MyCNN(nn.Module):
    def __init__(self, device, memory_depth=5, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05, Lambda=None):
        super(MyCNN, self).__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        bn1 = nn.BatchNorm2d(32)
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=2)

        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        bn2 = nn.BatchNorm2d(64)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d(kernel_size=2)

        fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        relu3 = nn.ReLU()
        self.cnn = nn.Sequential(conv1, bn1, relu1, pool1,
                                 conv2, bn2, relu2, pool2)
        self.fnn = nn.Sequential(fc1, relu3)
        self.final_layer = nn.Linear(in_features=128, out_features=10)
        self.apply(init_weights)
        self.device = device

        W = self.final_layer.weight.data.to(device)
        b = self.final_layer.bias.data.view(-1, 1).to(device)
        self.W = torch.cat([W, b], dim=1).to(device)

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
        
        # one epoch
        self.train_acc = []
        self.test_acc = []

    def forward(self, x, y=None, separate=False):
        feature_cnn = self.cnn(x).view(-1, 64 * 8 * 8)
        feature = self.fnn(feature_cnn)
        if separate:
            with torch.no_grad():
                ones_column = torch.ones((feature.shape[0], 1), device=self.device)
                Z = torch.cat((feature, ones_column), dim=1).T

                # get batch size
                n_calTk = x.shape[0]

                # store history
                if self.iter > self.memory_depth:
                    self.M = self.M[1:]  # remove oldest

                # add current batch to history
                self.M.append(Z)

                # solve for W and b
                self.solve(Z.to(self.device), y.T.to(self.device), n_calTk)

                # update regularization parameter and iteration
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist += [self.alpha]

                W = self.W.to(self.device)
                self.final_layer.weight.data = W[:, :-1]
                self.final_layer.bias.data = W[:, -1]
        return self.final_layer(feature)
        # return Phi @ self.W

    def solve(self, Z, C, n_calTk, dtype=torch.float32):
        M = torch.cat(self.M, dim=1)

        beta = 1.0
        if self.reduction == 'mean':
            beta = 1 / math.sqrt(n_calTk)

        W, info = tiksolvevec.solve(beta * Z, beta * C, beta * M, self.W, self.sumLambda, 
                                    device=self.device, Lambda=self.Lambda, dtype=dtype, opt_method=self.opt_method,
                                    lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W = W
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

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
        print('Slim_Kronecker Total training time = ', total_end - total_start)
        
        return self.train_acc, self.test_acc, total_end - total_start
    
    def train_one_epoch(self, train_loader, train_loader2, test_loader, optimizer, criterion, device, separate):
        k = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # one epoch
            self.train()
            criterion.reduction = 'mean'
            data, target = data.to(device), target.to(device)
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
        test_loss = 0
        correct = 0
        total = 0
        criterion.reduction = 'sum'
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
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

