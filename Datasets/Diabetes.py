import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

# 获取数据集
data = load_diabetes()

X = torch.from_numpy(data.data).float()
y = torch.from_numpy(data.target).float().view(-1, 1)

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
# y_train = torch.from_numpy(y_train).float()

X_val = torch.from_numpy(X_val).float()
# y_val = torch.from_numpy(y_val).float()

X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).float()

train = TensorDataset(X_train, y_train)
val = TensorDataset(X_val, y_val)
test = TensorDataset(X_test, y_test)
