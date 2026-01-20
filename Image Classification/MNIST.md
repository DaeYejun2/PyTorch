# MNIST 손글씨 이미지 분류

## 1. MNIST 데이터셋, 딥러닝의 'Hellow, World!' 🖐️
**MNIST(Modified National Institute of Standards and Technology)** 데이터셋은 0부터 9까지의 숫자가 손글씨로 쓰인 흑백 이미지 데이터셋이다. 각 이미지는 28x28 픽셀 크기이며,
총 60,000개의 학습 데이터와 10,000개의 테스트 데이터로 구성되어 있다.

## 2. PyTorch로 MNIST 분류
MLP 모델을 사용하여 MNIST 데이터셋을 분류. 선형 회귀와 마찬가지로 5단계를 따라 진행.
1. 데이터 준비: MNIST 데이터셋을 불러오고 전처리
2. 모델 설계: 입력, 은닉, 출력층으로 구성된 MLP 모델 정의
3. 손실 함수와 옵티마이저 정의: 분류 문제에 맞는 손실 함수와 옵티마이저 설정
4. 모델 학습: 준비된 데이터로 모델을 학습
5. 모델 평가: 학습된 모델의 성능(정확도)을 평가

## 1. 데이터 준비
PyTorch는 torchvision 라이브러리를 통해 MNIST 데이터셋을 매우 쉽게 불러올 수 있다. 데이터셋을 불러올 때, 텐서로 변환하고 정규화(Normalization)하는 변환(Transform)과정을 함께 적용한다.
```
import torch
import torchvision
import torchvision.transforms as transforms

# 텐서로 변환하고 -1~1 범위로 정규화하는 변환기 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # (평균, 표준편차)로 정규화
])

# 학습 데이터셋 및 데이터로더 불러오기
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 테스트 데이터셋 및 데이터로더 불러오기
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
```
DataLoader는 데이터셋을 미니 배치(mini-batch) 단위로 묶어 모델에 제공하는 역할을 한다.

## 2. 모델 설계
28x28 픽셀 이미지를 MLP의 입력으로 사용하기 위해, 이미지를 1차원 벡터로 펼쳐야 한다. 28*28 = 784 이므로, 입력층의 뉴런 수는 784개가 된다. 출력층은 0부터 9까지 10개의 클래스를 분류해야 하므로 10개의 뉴런을 가진다.
