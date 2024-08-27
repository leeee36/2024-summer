"""
데이터가 항상 머신러닝 알고리즘 학습에 필요한 형태로 제공되는 것이 아님.
변형을 해서 데이터를 조작하고 학습에 적합하게 만듦.
"""

"""
FashionMNIST 특징은 pil image 형식이며 정답은 정수이다.
학습을 하려면 정규화된 텐서 형태의 특징과 원핫으로 부호화된 텐서 형태의 정답이 필요.
이러한 변형을 위해 ToTensor , Lambda 사용
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

"""
ToTensor는 PIL image나 numpy ndarray를 floattensor로 변환하고,
이미지의 픽셀의 크기 값을 [0., 1.] 범위로 비례하여 조정함

Lambda는 사용자 정의 람다 함수를 적용함. 
여기에서는 정수를 원핫으로 부호화된 텐서로 바꾸는 함수를 정의함.
이 함수는 먼저 크기 10짜리 ZeroTensor를 만들고 scatter를 호출하여
주어진 정답 y에 해당하는 인덱스에 value=1을 할당함
"""
