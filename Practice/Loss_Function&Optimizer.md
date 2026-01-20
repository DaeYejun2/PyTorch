## 손실 함수와 옵티마이저

### 1. 손실함수(Loss Function)
손실 함수는 모델의 예측값과 실제 정답 사이의 오차 또는 불일치를 측정하는 함수이다. 손실 함수의 결과값, 즉 손실(Loss)이 클수록 모델의 성능이 나쁘다는 것을 의미한다.
<br>
딥러닝 모델의 학습 목표는 바로 이 손실 값을 최소화하는 것.

#### 대표적인 손실 함수 종류
손실 함수는 문제의 유형에 따라 달라진다.
* 회귀(Regression) 문제: 예측값이 연속적인 숫자일 때 사용
  MSE(Mean Squared Error): 예측값과 실제값의 차이를 제곱하여 평균을 낸 값. 오차의 크기를 강조하므로 이상치(outlier)에 민감하다.
* 분류(Classification) 문제: 예측값이 특정 클래스(범주)일 때 사용
  Cross-Entropy Loss(교차 엔트로피 손실): 모델이 예측한 확률 분포와 실제 정답의 확률 분포 사이의 차이를 측정한다. 분류 문제에서 가장 널리 사용되는 손실 함수
```
import torch.nn as nn

# 회귀 문제용: 평균 제곱 오차
mse_loss = nn.MSELoss()

# 분류 문제용: 교차 엔트로피 손실
ce_loss = nn.CrossEntropyLoss()
```

### 2. 옵티마이저(Optimizer)
손실 함수의 결과를 바탕으로 모델의 파라미터(가중치와 편향)를 업데이트하여 손실을 최소화하는 알고리즘. 경사 하강법을 사용하여 손실 함수의 기울기를 계산하고, 그 기울기의 반대 방향으로 파라미터를 업데이트한다.

#### 핵심 요소
1. 기울기 계산: loss.backward()를 호출하여 손실에 대한 모든 파라미터의 기울기를 계산한다.
2. 파라미터 업데이트: optimizer.step()을 호출하여 계산된 기울기를 바탕으로 파라미터의 값을 업데이트한다.
이때 optimizer.zero_grad()는 필수적인 단계. PyTorch는 미분값을 누적하기 때문에, 새로운 기울기를 계산하기 전에 이전 기울기를 0으로 초기화해야 한다.

#### 대표적인 옵티마이저 종류
가장 기본적인 옵티마이저는 SGD(확률적 경사 하강법)이며, 그 외에도 다양한 옵티마이저가 개발되었다.
* SGD: 간단하고 직관적이지만, 학습 속도가 느리거나 지역 최솟값에 빠지기 쉽다는 단점이 있다.
* Adam: 가장 널리 사용되는 옵티마이저 중 하나이다. 각 파라미터마다 학습률을 다르게 적용하는 적응형 학습률(Adaptive Learning Rate)방식을 사용해 SGD보다 더 빠르고 안정적인 학습을 가능하게 한다.
```
import torch.optim as optim

# 모델의 모든 학습 가능한 파라미터를 옵티마이저에게 전달
# SGD 옵티마이저
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam 옵티마이저
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)
```
학습률(lr, learning rate)은 옵티마이저가 파라미터를 얼마나 크게 업데이트할지를 결정하는 중요한 하이퍼파라미터이다.

### 3. 손실 함수와 옵티마이저의 조화 
* 손실 함수: 현재 모델의 상태를 평가하는 기준 제공
* 옵티마이저: 그 평가를 바탕으로 모델을 개선하는 방법 제공
```
for epochs in range(num_epochs):
  # 1. 순전파: 예측값 계산
  y_pred = model(x)

  # 2. 손실 계산: 예측값과 정답 간의 오차 측정
  loss = loss_fn(y_pred, y)
  
  # 3. 기울기 초기화: 이전 기울기 삭제
  optimizer.zero_grad()

  # 4. 역전파: 손실에 대한 기울기 계산
  loss.backward()

  # 5. 파라미터 업데이트: 기울기를 이용해 파라미터 조정
  optimizer.step()
```
**이 5단계는 앞으로 어떤 딥러닝 모델을 만들든 변하지 않는 기본적인 학습 과정이다**

### 4. 전체 코드
```
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 생성
# 실제 데이터의 특징: y = 2x + 1
w_true = 2.0
b_true = 1.0

X = torch.randn(100) * 10  # 100개의 x 데이터
Y = w_true * X + b_true + torch.randn(100) * 2  # y = 2x + 1 + 노이즈

# 2. 모델 정의
class LinearRegression(nn.Module):
  def __init__(self):
    super().__init__()
    ## nn.Linear는 w와 b를 자동으로 초기화해주는 모듈
    self.linear = nn.Linear(in_features=1, out_features=1)

  def forward(self, x):
    return self.linear(x.unsqueeze(-1)).squeeze(-1)

# 모델 객체 생성
model = LinearRegression()

# 3. 손실 함수 및 옵티마이저 설정
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 모델 학습 루프
num_epochs = 100

for epoch in range(num_epochs):
  # 1. 순전파: 예측값 계산
  y_pred = model(X)

  # 2. 손실 계산: 예측값과 정답 간의 오차 측정
  loss = loss_fn(y_pred, Y)
  
  # 3. 기울기 초기화: 이전 기울기 삭제
  optimizer.zero_grad()

  # 4. 역전파: 손실에 대한 기울기 계산
  loss.backward()

  # 5. 파라미터 업데이트: 기울기를 이용해 파라미터 조정
  optimizer.step()
  
  if (epoch+1) % 100 == 0:
    print(f'epoch [{epoch+1}/{num_epochs}], 손실: {loss.item():.4f}')

  print("\n--- 학습 완료 ---")
# 학습된 w와 b를 확인합니다.
print(f"학습된 w: {model.linear.weight.item():.4f}")
print(f"학습된 b: {model.linear.bias.item():.4f}")
print("-----------------")
print(f"실제 w: {w_true:.4f}")
print(f"실제 b: {b_true:.4f}")
```

| 구분 | 실제 값 (True) | 학습된 값 (Learned) |
| :--- | :--- | :--- |
| **Weight (w)** | 2.0 | 2.0245 |
| **Bias (b)** | 1.0 | 0.9523 |

학습된 w와 b의 값이 가정한 실제 w와 b 값에 매우 가깝다는 것을 알 수 있다.







