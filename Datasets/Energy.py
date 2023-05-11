import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

# 获取数据集
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
data = pd.read_excel(url)

# 分离输入和输出
X = data.iloc[:, :-2].values
y = data.iloc[:, -2:].values

# 划分训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=123)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 转为Tensor
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

train = TensorDataset(X_train, y_train)
val = TensorDataset(X_val, y_val)
test = TensorDataset(X_test, y_test)
