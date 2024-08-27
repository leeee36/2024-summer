""" pytorch regress"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ===== generating dataset ===== #
num_data = 2400
x1 = np.random.rand(num_data) * 10
x2 = np.random.rand(num_data) * 10
e = np.random.normal(0, 0.5, num_data)
X = np.array([x1, x2]).T
y = 2*np.sin(x1) + np.log(0.5*x2**2) + e

# ===== split dataset into train, val, test ===== #
train_X, train_y = X[:1600, :], y[:1600]
val_X, val_y = X[1600:2000, :], y[1600:2000]
test_X, test_y = X[2000:, :], y[2000:]

"""
# ===== visualize each dataset ===== #
fig = plt.figure(figsize=(12,5))

# fig : 하나의 큰 캔버스
# subplot : 캔버스 안에 각각의 그래프가 그려지는 곳
# subplot(row, col, 몇번째)

ax1 = fig.add_subplot(1,3,1, projection="3d")
ax1.scatter(train_X[:, 0], train_X[:, 1], train_y, c=train_y, cmap="jet")

ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.set_title('Train Set Distribution')
ax1.set_zlim(-10, 6)
ax1.view_init(40, -60)
ax1.invert_xaxis()


ax2 = fig.add_subplot(1,3,2, projection="3d")
ax2.scatter(val_X[:, 0], val_X[:, 1], val_y, c=val_y, cmap="jet")

ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title('Validation Set Distribution')
ax2.set_zlim(-10, 6)
ax2.view_init(40, -60)
ax2.invert_xaxis()


ax3 = fig.add_subplot(1,3,3, projection="3d")
ax3.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap="jet")

ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title('Test Set Distribution')
ax3.set_zlim(-10, 6)
ax3.view_init(40, -60)
ax3.invert_xaxis()

plt.show()
"""

# ===== pytorch ===== #

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x):
        # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값 리턴함
        return self.linear(x)
    
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
lm = LinearModel()

# ===== cost function define ===== #
reg_loss = nn.MSELoss()

"""
# uncomment for testing loss fn
test_pred_y = torch.Tensor([0,0,0,0])
test_true_y = torch.Tensor([0,1,0,1])
print(reg_loss(test_pred_y, test_true_y))
print(reg_loss(test_true_y, test_true_y))
"""

# ===== construct model ===== #
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

model = MLPModel()

# ===== construct optimizer ===== #
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr)

# 매 학습 단계에서의 epoch 값과 그 때의 loss 값을 저장할 리스트
list_epoch = []
list_train_loss = []
list_val_loss = []
list_mae = []
list_mae_epoch = []

epoch = 6000
for i in range(epoch):

    # ===== train ===== #
    model.train() # 모델을 학습 모드로 변경 / 이후 모델 평가 시 eval() 모드로 변경
    optimizer.zero_grad() # optimizer 0으로 초기화

    input_x = torch.Tensor(train_X)
    true_y = torch.Tensor(train_y)
    pred_y = model(input_x)
    # print(input_x.shape, true_y.shape, pred_y.shape)

    loss = reg_loss(pred_y.squeeze(), true_y)
    loss.backward()
    optimizer.step()
    list_epoch.append(i)
    list_train_loss.append(loss.item())

    # ===== validation ===== #
    model.eval()
    optimizer.zero_grad()
    input_x = torch.Tensor(val_X)
    true_y = torch.Tensor(val_y)
    pred_y = model(input_x)
    loss = reg_loss(pred_y.squeeze(), true_y)
    list_val_loss.append(loss.item())

    # ===== Evaluation ===== #
    if i % 200 == 0:
         
        # ===== calculate MAE ===== #
        model.eval()
        optimizer.zero_grad()
        input_x = torch.Tensor(test_X)
        true_y = torch.Tensor(test_y)
        pred_y = model(input_x).detach().numpy()
        mae = mean_absolute_error(true_y, pred_y)
        list_mae.append(mae)
        list_mae_epoch.append(i)

"""
5. Report Experiment
"""

fig = plt.figure(figsize=(15,5))

# ===== Loss Fluctuation ===== #
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_val_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_ylim(0, 5)
ax1.grid()
ax1.legend()
ax1.set_title('epoch vs loss')

# ===== Metric Fluctuation ===== #
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list_mae_epoch, list_mae, marker='x', label='mae metric')
ax2.set_xlabel('epoch')
ax2.set_ylabel('mae')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs mae')

plt.show()