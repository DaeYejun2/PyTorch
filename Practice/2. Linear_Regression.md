## 선형 회귀(Linear Regression)
선형 회귀는 한 개 이상의 독립 변수와 종속 변수 사이의 선형 관계를 모델링하는 통계 기법이다. 쉽게 말해, 데이터의 패턴을 가장 잘 나타내는 *직선*을 찾는 과정
<br>
예를 들어, "공부 시간"에 따른 "시험 점수"를 예측한다고 가정해 보자.
* 공부 시간(독립 변수): x
* 시험 점수(종속 변수): y
<br>
우리의 목표는 y = wx + b라는 직선의 방정식을 찾는 것이다.
* w: 기울기 (공부 시간이 늘어날스록 점수가 얼마나 오르는지)
* b: y절편 (공부를 하나도 안 했을 때의 기본 점수
딥러닝 관점에서 보면, w와 b는 모델이 학습해야 할 '파라미터'이다. 우리는 데이터(x와 y)를 통해 w와 b를 최적의 값으로 조정하여 가장 오차가 적은 직선을 찾아낼 것이다.

## PyTorch로 선형 회귀 모델 만들기
PyTorch로 딥러닝 모델을 만드는 과정은 크게 5단계로 나눌 수 있다.
1. 데이터 준비: 모델이 학습할 데이터를 만든다
2. 모델 설계: 모델의 구조를 정의한다
3. 손실 함수 및 옵티마이저 정의: 오차를 계산하고 파라미터를 업데이트할 방법을 설정한다
4. 모델 학습: 준비된 데이터를 이용해 모델을 반복적으로 학습시킨다
5. 모델 평가: 학습된 모델의 성능을 확인한다.

### 1. 데이터 준비
먼저 y = 2x + 1이라는 실제 직선을 가정하고, 이 직선 주변에 약간의 노이즈(오차)를 더한 가상의 데이터를 만든다.
<br>
```
import torch

#실제 데이터
w_true = 2.0
b_true = 1.0

#100개의 임의의 x값 생성
X = torch.randn(100)*10

#y값 생성: y = 2x+1 + 노이즈
Y = w_true * X + b_true + torch.randn(100)*2
```
실제 데이터처럼 오차가 존재하는 상황에서도 본질적인 패턴($w, b$)을 찾도록 하기 위해 의도적으로 노이즈를 추가

### 2. 모델 설계
PyTorch에서 모델을 정의할 때는 torch.nn.Module 클래스를 상속받아 사용한다. 이 클래스는 모델의 파라미터와 구조를 관리하는 데 필요한 기본 기능을 제공한다.
```
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
      super().__init__()
      # 학습할 파라미터 w와 b를 정의
      # nn.Parameter는 텐서를 모델의 파라미터로 등록하는 특별한 클래스
      self.w = nn.Parameter(torch.randn(1, requires_grad=True))
      self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
      # 모델의 순전파를 정의
      # 입력 x에 대해 예측값 y_pred를 계산
      return self.w * x + self.b

# 모델 객체 생성
model = LinearRegression()
```
__init__ 함수에서는 모델이 학습해야 할 파라미터(w, b)를 정의하고, forward 함수에서는 입력 x가 모델을 통과하여 최종 예측값 y_pred를 만드는 과정을 정의한다.

### 3. 손실 함수 및 옵티마이저 정의
* 손실함수(Loss Function): 모델의 예측값과 실제 정답 사이의 오차를 계산하는 함수. 선형 회귀에서는 주로 평균 제곱오차(Mean Squared Error, MSE)를 사용한다. PyTorch에서는 nn.MSELoss로 제공된다.
* 옵티마이저(Optimizer): 손실을 최소화하기 위해 모델의 파라미터(w, b)를 업데이트하는 알고리즘. 여기서는 확률적 경사 하강법(Stochastic Gradient Descent, SGD)을 사용한다. PyTorch에서는 torch.optim.SGD로 제공된다.
```
import torch.optim as optim

#손실 함수 정의
loss_fn = nn.MSELoss()

#옵티마이저 정의
#model.parameters()는 모델의 모든 학습 가능한 파라미터를 반환한다.
# lr (learnig rate): 한 번에 파라미터를 얼마나 크게 업데이트할지 결정하는 학습률
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### 4. 모델 학습
모델을 반복적으로 학습시키는 과정을 에포크(Epoch)라고 부른다. 한 번의 에포크는 전체 데이터를 한 바퀴 도는 것을 의미
```
num_epochs = 1000

for epoch in range(num_epochs):
  #1. 순전파
  y_pred = model(X)

  #2. 손실 계산
  loss = loss_fn(y_pred, Y)

  #3. 역전파
  #옵티마이저에 저장된 이전 기울기를 초기화
  optimizer.zero_grad()
  #손실에 대한 모든 파라미터의 미분값(기울기)을 계산
  loss.backward()

  #4. 파라미터 업데이트
  #계산된 기울기를 사용하여 파라미터를 업데이트
  optimizer.step()

  # 100 에포크마다 손실 출력
  if (epoch+1) % 100 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# Epoch [100/1000], Loss: 4.0750
# Epoch [200/1000], Loss: 4.0516
# Epoch [300/1000], Loss: 4.0511
# Epoch [400/1000], Loss: 4.0511
# Epoch [500/1000], Loss: 4.0511
# Epoch [600/1000], Loss: 4.0511
# Epoch [700/1000], Loss: 4.0511
# Epoch [800/1000], Loss: 4.0511
# Epoch [900/1000], Loss: 4.0511
# Epoch [1000/1000], Loss: 4.0511
```
### 5. 모델 평가
학습이 끝난 후, 모델의 최종 파라미터(w, b)와 실제 파라미터(w_true, b_true)를 비교하여 모델이 잘 학습되었는지 확인
```
# 학습된 모델의 최종 파라미터
print("\n--- 최종 파라미터 ---")
print(f"학습된 w: {model.w.item():.4f}")
print(f"학습된 b: {model.b.item():.4f}")
print("---------------------")
print(f"실제 w: {w_true}")
print(f"실제 b: {b_true}")
```
| 구분 | 실제 값 (True) | 학습된 값 (Learned) |
| :--- | :--- | :--- |
| **Weight (w)** | 2.0 | 2.0069 |
| **Bias (b)** | 1.0 | 0.9225 |

### 6. 학습 결과 (Training Result)
데이터에 노이즈를 추가하여 실제 환경과 유사하게 만들었음에도 불구하고, 모델이 정답에 가깝게 수렴했다.

> **결론:** 무작위로 초기화된 파라미터들이 backward()를 통한 Gradient Descent(경사 하강법) 과정을 거쳐 최적의 값으로 업데이트되었습니다.
