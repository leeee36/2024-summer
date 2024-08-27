"""
신경망 모델 구성하기

신경망은 데이터에 대한 연산을 수행하는 계층/모듈 로 구성되어 있음.
torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소 제공함.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 학습을 위한 장치 얻기

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device : {device}")

# 클래스 정의하기
"""
신경망 모델을 nn.Module의 하위클래스로 정의하고, init에서 신경망 계층들 초기화함
nn.Module 을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산을 구현함.
"""

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
model = NeuralNetwork().to(device)
print(model)

print("---------------------------------------------------------")

# 모델을 사용하기 위해 입력 데이터를 전달함.
# model.forward() 를 직접 호출하지 말 것

"""
모델에 입력을 전달하여 호출하면 2차원 텐서를 반환함
2차원 텐서의 dim=0은 각 분류에 대한 원시 예측값 10개,
dim=1에는 각 출력의 개별 값들이 해당함. 
원시 예측값을 nn.Softmax 모듈의 인스턴스에 통과시켜 예측 확률을 얻음.
"""

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

print("---------------------------------------------------------")

# 모델 계층

input_image = torch.rand(3, 28, 28)
print(input_image.size())

#nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image) #2d image, 784 pixels
print(flat_image.size())

#nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print("hidden: ", hidden1.size())

#nn.ReLU
print(f"Before ReLU : {hidden1} \n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

#nn.Sequential
#순서를 가지는 모듈의 컨테이너.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

#nn.softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print("------------------------------------------------------------")

#model parameter
#아래 예제에서는 각 매개변수를 순회하며 매개변수의 크기와 값을 출력함
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
