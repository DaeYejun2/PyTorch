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
```
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    # 입력층: 784
    self.fc1 = nn.Linear(784, 128)
    # 은닉층
    self.fc2 = nn.Linear(128, 64)
    # 출력층: 10 (0~9)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    # 이미지를 1차원 벡터로 펼치기 (flatten)
    x = x.view(-1, 28*28)
    # 순전파
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)  # 출력층에는 활성화 함수를 사용하지 않는다
    return x
```
view() 함수는 텐서의 모양을 변경하는 데 사용되며, 여기서는 28x28 이미지를 784개의 원소를 가진 1차원 벡터롤 변환한다.

## 3 & 4. 손실 함수, 옵티마이저 정의 및 학습
분류 문제에서는 주로 CrossEntropyLoss를 사용. 옵티마이저는 Adam 사용
```
# 모델, 손실 함수, 옵티마이저 정의
model = MLP()
criterion = nn.CrossEntropyLoss()  # 손실함수. 모델의 예측과 정답 사이의 거리 측정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 미분값을 보고 실제로 가중치를 어떻게 수정할지 결정하는 알고리즘

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
  running_loss = 0.0
  for i, data in enumerate(train_loader):
    inputs, labels = data

    # 순전파
    outputs = model(inputs)
    loss = criterion(outputs, labels)  # (예측값, 정답) 으로 손실값 계산

    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f'에포크 [{epoch+1}/{num_epochs}], 손실: {running_loss/len(train_loader):.4f}')
print("학습 완료!")
```

## 5. 모델 평가
학습된 모델의 성능을 테스트 데이터셋으로 평가. 정확도(Accuracy)를 계산
```
correct = 0
total = 0
with torch.no_grad():
  for data in test_loader:
    images, labels = data
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1) # 모델이 각 숫자(0~9)일 확률을 여러 개 내놓으면, 그중 가장 높은 확률을 가진 인덱스를 모델의 최종 답변(예측값)으로 선택
    total += labels.size(0)
    correct += (predicted == labels).sum().item()  # 모델의 답변과 실제 정답이 일치하는 개수를 세어 전체 데이터 대비 백분율 확인

print(f'\n테스트 데이터셋에서 모델의 정확도: {100 * correct / total:.2f}%')
```


### Bonus
1. batch_size 64->256
   한 번에 처리하는 데이터 양이 많아져 학습 속도는 빨라지지만, 가중치 업데이트 횟수가 줄어들어 정확도가 낮아질 수 있다,
2. MLP 클래스 은닉층 수를 하나 더 추가하거나, 은닉층의 뉴런 수 변경
   모델이 복잡해질수록(깊어질수록) 데이터의 미세한 패턴을 더 잘 찾아낼 수 있어 정확도가 상승하지만, 연산량이 늘어나 학습 시간도 증가한다.
3. optimizer 를 torch.optim.SGD로 변경했을 때 학습률(lr)을 어떻게 조절해야 하는지
   *SGD (lr=0.001): 정확도 67.25% (학습이 거의 안 됨)
   *SGD (lr=0.1): 정확도 99.0% (매우 잘 됨)
    Adam은 학습률을 스스로 조절하는 기능이 있어 0.001에서도 잘 작동하지만, SGD는 고정된 학습률을 사용하므로 훨씬 큰 학습률(예: 0.1)을 설정해야 제 성능을 발휘
