import torch.nn as nn
import argparse
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from multiprocessing import freeze_support


# ===== Preparing Dataset ===== #

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform) # files already downloaded and verified 안내문자 False 지정 시 미출력

# loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ===== Model ===== #

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, n_layer, act):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.act = act

        self.fc = nn.Linear(self.in_dim, self.hid_dim)
        self.linears = nn.ModuleList()

        for i in range(self.n_layer-1):
            self.linears.append(nn.Linear(self.hid_dim, self.hid_dim))
        
        self.fc2 = nn.Linear(self.hid_dim, self.out_dim)

        if self.act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc(x))
        for fc in self.linears:
            x = self.act(fc(x))
        x = self.fc2(x)

        return x


# ===== hyperparameters ===== #
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

                    # setting value & # finding value
args.in_dim = 3072  # 3072            #
args.out_dim = 10   # 10              #
args.hid_dim = 1024 # 100             # 1024
args.n_layer = 2    # 5               # 2
args.act = 'relu'   # relu            #
args.lr = 0.0002    # 0.001           # 0.0001
args.mm = 0.9       # 0.9             #
args.epoch = 15     # 2               # 15
args.p = 0.2                          # 0.2

# print(args)
device = torch.device("mps")

def experiment(args):
    net = MLP(args.in_dim, args.out_dim, args.hid_dim, args.n_layer, args.act)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mm) 
    # drop = nn.Dropout(p=args.p)

    for epoch in range(args.epoch):  # loop over the dataset multiple times

        # ===== Train ===== #
        running_loss = 0.0
        train_loss = 0.0
         
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, 3072)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss += loss.item()
            
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0
        
        train_loss = train_loss / len(trainloader)

        # ===== Validation ===== #
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.view(-1, 3072)
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)

                # val loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = correct / total * 100
        
        print('Epoch {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, train_loss, val_loss, val_acc))

    
    # ===== Evaluation ===== #
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(-1, 3072)
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = correct / total * 100
    
    print("==============================")
    print('Test Acc: {}'.format(test_acc))
    
    print("=================")
    print('Finished Training')
    print("=================================================================")

if __name__ == "__main__":
    
    freeze_support()
    # experiment(args)

    list_var1 = [2, 4, 8]
    list_var2 = [512, 1024, 2048]

    for var1 in list_var1:
        for var2 in list_var2:
            print(f'# ===== The result of \'n_layer: {var1}\' & \'hid_dim: {var2}\' ===== #')
            args.n_layer = var1
            args.hid_dim = var2
            result = experiment(args)

"""
[n_layer & hid_dim]
n_layer 수가 높다고 하여 좋은 것은 아님 

: n_layel , hid_dim = 2 , 1024 / val_acc = '49.04'
                    = 2 , 128  / val_acc = '48.xx'

[lr & epoch]
: lr , epoch = 0.0001 , 15 / val_acc = '54.47'

"""