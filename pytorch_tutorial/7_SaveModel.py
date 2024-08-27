# 모델 저장하고 불러오기

import torch
import torchvision.models as models

# 모델 가중치 저장하고 불러오기
# 학습한 매개변수는 state_dict 라고 불리는 internal state dictionary에 저장됨

model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")

# 모델 가중치 불러오기

model = models.vgg16() # 학습되지 않은 모델 생성
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# -------------------------------------------------------------------

# 모델의 형태를 포함하여 저장하고 불러오기

torch.save(model, "model.pth")
model = torch.load("model.pth")
