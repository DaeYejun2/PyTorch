# Cats vs Dogs

## 1. 데이터셋 준비
```
!pip install kagglehub
import kagglehub
path = kagglehub.dataset_download("biaiscience/dogs-vs-cats")
```
## 2. 데이터 전처리 및 증강
* 크기 통일: 사전 훈련 모델(예: ResNet)은 특정 크기(예: 224x224)의 이미지를 입력으로 받도록 설계되어있다. 따라서 모든 이미지를 이 크기로 통일 해야함
* 정규화: 대부분의 딥러닝 모델은 픽셀 값이 0 ~ 1 또는 -1 ~ 1 사이인 정규화된 입력값을 선호한다. ImageNet으로 학습된 모델은 ImageNet 평균과 표준편차로 정규화 하는 것이 일반적이다.
```
import torchvision.transforms as transforms

# 학습 데이터에 적용할 변환 및 증강
train_transforms = transforms.Compose([
    transforms.RandomRotation(10), # 이미지를 무작위로 10도 회전
    transforms.RandomResizedCrop(224), # 무작위로 자른 후 224x224로
    transforms.RandomHorizontalFlip(), # 50% 확률로 좌우 반전
    transforms.ToTensor(), # 이미지를 텐서로 변환
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 평균
                         [0.229, 0.224, 0.225])  # ImageNet 표준편차
])

# 검증 및 테스트 데이터에 적용할 변환
# 평가용 데이터에는 증강을 적용하지 않는다.
valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```
## 3. 모델 설계 및 전이 학습
ImageNet으로 이미 학습된 ResNet-18 모델을 활용한 전이 학습
<br>
**ResNet**은 잔차 블록(Residual Block)을 통해 수백 개 이상의 깊은 층을 효율적으로 학습할 수 있게 만든 CNN 모델

1. 사전 훈련 모델 불러오기
2. 가중치 고정: 모델의 모든 파라미터의 required_grad를 False로 설정하여, 이 파라미터들이 학습 중에 업데이트되지 않도록 한다
3. 분류기 교체: 마지막 완전 연결 층(Fully Connected Layer)인 fc층을 우리의 문제(고양이vs강아지)에 맞게 새로 만든다.

## 4. 통합 코드
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import shutil

# 실제 이미지들이 있는 경로
base_path = '/kaggle/input/dogs-vs-cats/train/train' # 또는 실제 데이터 경로
# 쓰기 권한이 있는 새로운 경로 (캐글의 경우 /kaggle/working 이용)
working_dir = '/kaggle/working/data'
train_dir = os.path.join(working_dir, 'train')
cat_dir = os.path.join(train_dir, 'cat')
dog_dir = os.path.join(train_dir, 'dog')

# 폴더 생성
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

# 파일 이동 (원본 주소에서 새 주소로)
files = os.listdir(base_path)
for f in files:
    if f.lower().startswith('cat'):
        shutil.copy(os.path.join(base_path, f), os.path.join(cat_dir, f))
    elif f.lower().startswith('dog'):
        shutil.copy(os.path.join(base_path, f), os.path.join(dog_dir, f))

print(f"정리 완료! 고양이: {len(os.listdir(cat_dir))}장, 강아지: {len(os.listdir(dog_dir))}장")

# TRAIN_DIR = os.path.join(path, 'train') # 데이터셋이 있는 폴더 경로
TRAIN_DIR = '/kaggle/working/data/train'
MODEL_SAVE_PATH = './cat_dog_classifier.pth' # 학습된 모델을 저장할 경로

# 학습 데이터에 적용할 변환 및 증강
train_transforms = transforms.Compose([
    transforms.RandomRotation(10), # 이미지를 무작위로 10도 회전
    transforms.RandomResizedCrop(224), # 무작위로 자른 후 224x224로
    transforms.RandomHorizontalFlip(), # 50% 확률로 좌우 반전
    transforms.ToTensor(), # 이미지를 텐서로 변환
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 평균
                         [0.229, 0.224, 0.225])  # ImageNet 표준편차
])

# 검증 및 테스트 데이터에 적용할 변환
# 평가용 데이터에는 증강을 적용하지 않는다.
valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ImageFolder는 폴더 구조를 기반으로 자동으로 레이블 지정
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=valid_transforms)

indices = list(range(len(train_dataset)))
split = int(0.2 * len(train_dataset)) # 20%는 검증용으로

np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_subset = Subset(train_dataset, train_idx)
valid_subset = Subset(test_dataset, valid_idx)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=2)

# 전이 학습을 위한 ResNet18
# ImageNet으로 사전 훈련된 모델 불러오기
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

# 모델의 모든 파라미터(특징 추출기)를 동결 (=학습하지 않음)
for param in model.parameters():
    param.requires_grad = False

# 마지막 완전 연결 층을 현재 문제에 맞게 수정
# ResNet-18의 마지막 층은 'fc'
num_ftrs = model.fc.in_features # 마지막 층의 입력 뉴런수
model.fc = nn.Linear(num_ftrs, 2) # 2개의 클래스(고양이, 강아지)로 변경

# 모델을 GPU로
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# 학습할 파라미터는 새로 만든 'fc' 층의 파라미터만
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
since = time.time() # 학습 시작 시간

for phase in ['train', 'valid']:
  if phase == 'train':
    model.train() # 모델을 학습 모드로 설정
    dataloader = train_loader
  else:
    model.eval() # 모델을 평가 모드로 설정
    dataloader = valid_loader

  running_loss = 0.0
  running_corrects = 0

  # 데이터 로더에서 미니 배치 단위로 데이터 가져오기
  for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
      # 학습 단계에서만 기울기 계산
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1) # 가장 높은 확률의 클래스 예측
      loss = criterion(outputs, labels)

      # 학습 단계에서만 역전파 및 파라미터 업데이트
      if phase == 'train':
        loss.backward()
        optimizer.step()

    # 통계 계산
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

  # epoch당 손실 및 정확도 계산
  epoch_loss = running_loss / len(dataloader.dataset)
  epoch_acc = running_corrects.double() / len(dataloader.dataset)

  print(f'{phase} loss: {epoch_loss:.4f} | {phase} accuracy: {epoch_acc:.4f}')

# 전체 학습 소요 시간
time_elapsed = time.time() - since
print(f'\n학습 완료| 총 소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초')

# 모델 저장
torch.save(model.state_dict(), MODEL_SAVE_PATH)
```
파일에 사진으로 들어 있어서 뺑이 좀 친거 같다..
<br>
* 정리 완료! 고양이: 12500장, 강아지: 12500장
* train loss: 0.2050 | train accuracy: 0.9122
* valid loss: 0.0575 | valid accuracy: 0.9798

* 학습 완료| 총 소요 시간: 1분 35초
### 핵심 구조
1. train_transforms: 이미지 크기 맞추고 데이터 증강
2. Subset: 학습용, 시험용 데이터 8:2
3. ResNet-18: 사전 훈련 모델
4. param.requires_grad = False: 기존 features는 고정하고 마지막 층 수만 개/고양이 판단 기준만 추가 학습

## 5. 학습된 모델로 새로운 이미지 분류
### 1. 모델 불러오기
```
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# 1. 모델 아키텍처 정의 (저장할 때와 동일해야 함)
model = torchvision.models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 2. 저장된 모델의 상태 사전 불러오기
MODEL_SAVE_PATH = './cat_dog_classifier.pth'
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# 3. 모델을 평가 모드로 전환
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
### 2. 이미지 전처리
```
def preprocess_image(image_path):
  # PIL(Pillow)를 사용해 이미지 로드
  image = Image.open(image_path).convert('RGB')

  # 학습에 사용했던 변환과 동일하게 적용 (단, 증강은 제외)
  transform = transform.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transform.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
  ])

  # 전처리된 이미지 텐서 변환 (배치 차원 추가)
  return transform(image).unsqueeze(0)
```
### 3. 예측 수행
```
def predict_image(model, image_path):
  image_tensor = preprocess_image(image_path).to(device)

  # 예측 수행
  with torch.no_grad():
    output = model(image_tensor)
    # 클래스별 확률 계산
    probabilities = nn.functional.softmax(output, dim=1)[0]

  # 가장 높은 확률의 클래스 인덱스 및 확률 추출
  _, predicted_class_idx = torch.max(output, 1)

  # 클래스 레이블 매핑 (ImageFolder의 클래스 인덱스 순서와 동일)
  class_labels = {0: 'Cat', 1: 'Dog'}
  predicted_label = class_labels[predicted_class_idx.item()]

  return predicted_label, probabilities.cpu().numpy()

# 예측을 원하는 이미지 경로
image_to_predict = '/kaggle/input/dogs-vs-cats/test/test/7981.jpg'

if os.path.exists(image_to_predict):
    prediction, probs = predict_image(model, image_to_predict)
    print(f"이미지 '{image_to_predict}'의 예측 결과: {prediction}")
    print(f"확률: 고양이 {probs[0]:.4f}, 강아지 {probs[1]:.4f}")
else:
    print("예측할 이미지를 찾을 수 없습니다. 경로를 확인해 주세요.")
```
## 6. 시각화
```
import matplotlib.pyplot as plt
import random

plt.figure(figsize=(20, 10))
for i in range(5):
    # 위 로직을 반복하여 이미지 경로 가져오기
    cat_or_dog = random.choice(['cat', 'dog'])
    img_name = random.choice(os.listdir(f'/kaggle/working/data/train/{cat_or_dog}'))
    path = f'/kaggle/working/data/train/{cat_or_dog}/{img_name}'
    
    pred, prob = predict_image(model, path)
    
    # 시각화
    plt.subplot(1, 5, i+1)
    plt.imshow(Image.open(path))
    plt.title(f"Target: {cat_or_dog}\nPred: {pred}\n({max(prob)*100:.1f}%)", fontsize=14)
    plt.axis('off')
plt.show()
```
<img width="1570" height="490" alt="image" src="https://github.com/user-attachments/assets/1690d2bd-1a8e-45f2-829b-68fd128961b5" />
