"""
모델 매개변수 최적화하기

모델과 데이터 준비 완료. 데이터에 매개변수를 최적화하여 모델을 학습하고 검증하고 테스트할 치례.
모델을 학습하는 과정은 반복적인 과정을 거친다.
각 반복 단꼐에서 모델은 출력을 추측하고, 추측과 정답 사이의 손실을 계산하고
매개변수에 대한 오류의 도함수를 수집한 뒤, 경사하강법을 사용하여 이 파라미터들을 최적화 한다.
"""

# 기본 코드

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

model = NeuralNetwork()

"""
하이퍼 파라미터

에포크 수 = 데이터셋을 반복하는 횟수
배치 크기 = 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
학습률 = 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 
    값이 작을수록 학습 속도가 느려지고, 
    값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있음.
"""

learning_rate = 1e-3
batch_size = 64
epochs = 5

"""
최적화 단계

하이퍼파라미터 설정 후에는 최적화 단계를 통해 모델을 학습하고 최적화할 수 있다.
최적화 단계의 각 반복을 에폭이라고 부름.

하나의 에폭은 두 부분으로 구성됨
1. 학습 단계(training loop) - 학습용 데이터셋을 반복하고 최적의 매개변수로 수렴함
2. 검증/테스트 단계(validation/test loop) - 모델 성능이 개선되고 있는지를 확인하기 위해
테스트 데이터셋을 반복함
"""

# -----------------------------------------------------------------

"""
손실 함수 (loss function)

손실함수는 획득한 결과와 실제 값 사이의 틀린 정도를 측정함
학습 중에 이 값을 최소화 하려고 함
주어진 데이터 샘플을 입력으로 계산한 예측과 정답을 비교하여 손실을 계산함!
"""

# 손실 함수 초기화
# 모델의 출력 logit을 CrossEntropyLoss에 전달하여 logit을 정규화 하고 예측 오류 계산함

loss_fn = nn.CrossEntropyLoss()

# -----------------------------------------------------------------

"""
옵티마이저 (optimizer)

# 최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정이다. #
최적화 알고리즘은 이 과정이 수행되는 방식을 정의한다.
모든 최적화 절차는 optimizer 객체에 캡슐화 됨
"""

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

"""
학습 단계(loop)에서 최적화는 세 단계로 이뤄짐
1. optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정한다.
    기본적으로 변화도는 더해짐 따라서 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정
2. loss.backwards()를 호출하여 예측 손실을 역전파한다.
    pytorch는 각 매개변수에 대한 손실의 변화도를 저장함
3. 변화도를 계산한 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로
    매개변수를 조정함.
"""

# -----------------------------------------------------------------

# 전체 구현

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # 모델을 평가 모드로 설정
    # 배치 정규화 및 드롭아웃 레이어에 중요함 ,,
    # 이 예시에서는 없어도 되지만 모범 사례를 위해 추가해둠
    model.eval() 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
