"""
데이터 샘플을 처리하는 코드는 지저분하고 유지보수가 어려울 수 있다.
더 나은 가독성과 모듈성을 위해 데이터셋 코드를 모델 학습 코드로부터 분리하는 것이 이상적.
pytorch는 데이터셋과 데이터로더 두 가지 데이터 기본 요소를 제공함.
Dataset : 샘플과 정답을 저장하고
Dataloader : dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체로 감싼다.

"""

# 데이터셋 불러오기

"""
root : 학습/테스트 데이터가 저장되는 경로
train : 학습용 또는 테스트용 데이터셋 여부를 지정
download=True : 루트에 데이터가 없는 경우 인터넷에서 다운로드함
transform과 target_transform은 특징과 정답 변형을 지정함

"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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

"""
데이터셋 순회하고 시각화하기
dataset에 리스트처럼 직접 접근할 수 있음
"""

labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

"""
파일에서 사용자 정의 데이터셋 만들기

사용자 정의 dataset 클래스는 반드시 3개 함수를 구현해야 함
__init__, __len__, __getitem__

FashionMNIST 이미지들은 img_dir 디렉토리에 저장되고, 
정답은 annotations_file csv 파일에 별도로 저장됨
"""

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            return image, label
        
"""
__init__ 함수는 객체가 생성될 때 한 번만 실행된다.
여기서는 이미지와 주석파일(annotation_file)이 포함된 디렉토리와 두가지 transform을 초기화함

__len__ 함수는 데이터셋의 샘플 개수를 반환한다.

__getitem__ 함수는 주어진 인덱스에 해당하는 샘플을 데이터셋에서 불러오고 반환한다.
인덱스를 기반으로 디스크에서 이미지 위치를 식별하고,
이미지를 텐서로 변환하고,
csv데이터로부터 해당하는 정답을 가져오고,
해당하는 경우 변형함수들을 호출한 뒤 텐서 이미지와 라벨을 사전형으로 반환함.
"""


"""
DataLoader로 학습용 데이터 준비하기

dataset은 데이터셋의 특징을 가져오고 하나의 샘플에 정답을 지정하는 일을 한 번에 함.
모델을 학습할 때, 일반적으로 샘플들을 미니배치로 전달하고, 
매 에포크마다 데이터를 다시 섞어서 과적합을 막고,
파이썬의 multiprocessing을 사용하여 데이터 검색 속도를 높이려고 함.

"""

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# DataLoader를 통해 순회하기(iterate)
# 순회 : 여러 곳을 돌아다니다

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
