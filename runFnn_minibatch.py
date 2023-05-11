import pickle
import torch
import random
# import SlimTrainFnn as Net
import torch.optim as optim
# from Datasets import Diabetes
from Datasets import Energy
train_data = Energy.train
test_data = Energy.test
"""# SepSA and Adam, RMS, SGD, NAG import this."""
import MyFNN_minibatch as Net
"""# slimTrain import this"""
# import SlimTrainFnn_Kronecker_minibatch as Net

# train_loss = []
# test_loss = []
# total_t = []
result = {}

# for seed in range(200, 300):
for lr in [1e-2, 1e-3, 1e-4]:
    seed = 233
    torch.manual_seed(seed)
    random.seed(seed)

    net = Net.MyNetwork(input_size=train_data[:][0].shape[1], out_width=train_data[:][1].shape[1])

    optimizer_SepSA = optim.RMSprop([{'params': net.feature_extractor.parameters(), 'weight_decay': 1e-5}],
                               lr=lr, momentum=0.9)
    optimizer_slimTrain = optim.Adam([{'params': net.feature_extractor.parameters(), 'weight_decay': 1e-5}],
                                 lr=lr)
    optimizer_Adam = optim.Adam([{'params': net.parameters(), 'weight_decay': 1e-5}],
                                lr=lr)
    optimizer_SGD = optim.SGD([{'params': net.parameters(), 'weight_decay': 1e-5}],
                              lr=lr)
    optimizer_Nest = optim.SGD([{'params': net.parameters(), 'weight_decay': 1e-5}],
                               lr=lr, nesterov=True, momentum=0.9)
    optimizer_RMS = optim.RMSprop([{'params': net.parameters(), 'weight_decay': 1e-5}],
                                  lr=lr, momentum=0.9)

    rls_train, rls_test, total_time = net.sgd_train(train_data, test_data, optimizer_SepSA, epochs=50, batch_size=32, separate=True)
    # rls_train, rls_test, total_time = net.sgd_train(train_data, test_data, optimizer_slimTrain, epochs=20, batch_size=32, separate=True)
    # rls_train, rls_test, total_time = net.sgd_train(train_data, test_data, optimizer_Adam, epochs=20, batch_size=32, separate=False)

    result.update({f'train{lr}': rls_train, f'test{lr}': rls_test})
with open('Result/Energy/mini-batch/SepSA.pickle', 'wb') as f:
    pickle.dump(result, f)
