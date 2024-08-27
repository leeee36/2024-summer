"""
MLP model 직접 구현하여 조건 변경해보며
MNIST dataset Accuracy 확인해보기
"""

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import accuracy_score
import time

batch_size = 128
train_dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_dataset =  datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
# print(len(train_dataset), len(val_dataset), len(test_dataset)) # 50000 10000 10000

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# ===== Model Architecture ===== #

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(784, 10, bias=True)
    
    def forward(self, x):
        x = self.linear(x)
        return x
    
class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.activ = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.activ(x)
        return x

# ===== Cost Function ===== #

cls_loss = nn.CrossEntropyLoss()

# ===== Train & eval ===== #

# ===== Model ===== #

model = MLPModel(784, 10, 100) # 입력 데이터의 크기는 28 * 28 = 784
device = 'mps' if torch.cuda.is_available() else 'cpu'
# print(torch.backends.mps.is_available()) # mps 작동여부 확인
model.to(device)
# ===== Optimizer ===== #

lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr)

list_epoch = [] 
list_train_loss = []
list_val_loss = []
list_acc = []
list_acc_epoch = []

epoch = 30
for i in range(epoch):

    ts = time.time()
    
    # ====== Train ====== #
    train_loss = 0
    model.train() 
    
    for input_X, true_y in train_loader:
        optimizer.zero_grad()

        input_X = input_X.squeeze()
        input_X = input_X.view(-1, 784)
        input_X = input_X.to(device)
        true_y = true_y.to(device)
        pred_y = model(input_X)

        loss = cls_loss(pred_y.squeeze(), true_y)
        loss.backward() 
        optimizer.step() 
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)
    list_train_loss.append(train_loss)
    list_epoch.append(i)
    
    
    # ====== Validation ====== #
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for input_X, true_y in val_loader:
            input_X = input_X.squeeze()
            input_X = input_X.view(-1, 784)
            input_X = input_X.to(device)
            true_y = true_y.to(device)
            pred_y = model(input_X)

            loss = cls_loss(pred_y.squeeze(), true_y)
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        list_val_loss.append(val_loss)


    # ====== Evaluation ======= #
    correct = 0
    model.eval()

    with torch.no_grad():
        for input_X, true_y in test_loader:
            input_X = input_X.squeeze()
            input_X = input_X.view(-1, 784)
            input_X = input_X.to(device)
            true_y = true_y.to(device)
            pred_y = model(input_X).max(1, keepdim=True)[1].squeeze()
            correct += pred_y.eq(true_y).sum()

        acc = correct.numpy() / len(test_loader.dataset)
        list_acc.append(acc)
        list_acc_epoch.append(i)
    
    te = time.time()

    print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Test Acc: {}%, time: {:3.1f}'.format(i, train_loss, val_loss, acc*100, te-ts))

# ===== Report Experiment ===== #

fig = plt.figure(figsize=(15,5))

# ====== Loss Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_val_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.grid()
ax1.legend()
ax1.set_title('epoch vs loss')

# ====== Metric Fluctuation ====== #
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list_acc_epoch, list_acc, marker='x', label='Accuracy metric')
ax2.set_xlabel('epoch')
ax2.set_ylabel('Acc')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs Accuracy')

plt.show()