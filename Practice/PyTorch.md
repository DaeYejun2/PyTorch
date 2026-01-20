## Tensor
**숫자를 담는 다차원 배열.** > 행렬(Matrix)을 임의의 차원으로 일반화한 수학적 개념입니다.

## PyTorch & Numpy
### 공통점
* 두 라이브러리 모두 다차원 배열(Multi-dimensional Array)을 효율적으로 다루기 위해 설계되었습니다.
### 차이점
#### 1) GPU 연산 활용
* PyTorch는 GPU를 활용하여 연산 속도를 크게 높일 수 있다.(Numpy는 CPU만을 사용하여 연산)

#### 2) 자동 미분 (Autograd)
* 텐서의 모든 연산에 대한 미분 값을 자동으로 계산해 줍니다. 이는 딥러닝 모델의 핵심인 경사 하강법(Gradient Descent)을 구현할 때 사용된다.

## Autograd
### 왜 미분이 필요한가
* 딥러닝 모델은 학습 과정에서 수많은 데이터를 통해 정답에 가까워진다. 이 과정은 마치 산 정상에 올라가는 것과 같다. 산의 가장 낮은 지점(최소 손실)을 찾아 내려가려면, 현재 서 있는 위치에서 어느 방향으로 가야 경사가 가장 가파른지 알아야 한다. 이 '방향'과 '경사도'를 수학적으로 계산하는 것이 바로 **미분**이다
* 딥러닝에서는 이 경사도를 기울기(Gradient)라고 부른다. 모델은 이 기울기를 이용해 파라미터(가중치와 편향)를 조금씩 조정하면서 손실(Loss)을 최소화한다. 이 과정을 경사 하강법(Gradient Descent)이라고 한다.

### Autograd의 역할
* 텐서에 어떤 연산을 수행하든, 그 과정을 모두 기록하고 나중에 이 기록을 바탕으로 미분값을 자동으로 계산한다,
##### requires_grad = True
```
import torch

# x텐서의 미분값을 계산할 수 있도록 설정
x = torch.tensor(2.0, requires_grad=True)

#y는 x를 이용한 연산이므로 자동으로 미분 가능
y = x**2
z = y * 3

print(z)
print(z.requires_grad) #True
```

* z가 계산 그래프의 일부라는 것을 나타낸다

### backward() 메서드로 미분값 계산하기
*최종 결과 텐서(z)에 대해 .backward() 메서드를 호출하면, PyTorch는 계산 그래프를 역전파하며 각 텐서의 미분값을 자동으로 계산한다. 미분값은 .grad 속성에 저장된다.

```
import torch

# x텐서의 미분값을 계산할 수 있도록 설정
x = torch.tensor(2.0, requires_grad=True)

#y는 x를 이용한 연산이므로 자동으로 미분 가능
y = x**2
z = y * 3

#z를 x에 대해 미분
#dz/dx = 3 * (2x) = 6x
#x=2 이므로, dz/dx = 12
z.backward()

#x의 미분값(기울기) 확인
#z에 대한 x의 기울기(gradient)
print(f'x의 미분값(dz/dx): {x.grad}')
# 출력: x의 미분값(dz/dx): 12.0
```
* $3x^2$을 미분한 값 6x에 x=2.0 대입 -> 12.0

### 경사 하강법
* y=wx + b 모델 학습

```
import torch

#학습할 가중치(w)와 편향(b) 정의
#이 두 텐서의 미분값을 계산해야 하므로 requires_grad=True 설정
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

#입력 데이터와 정답 데이터
x = torch.tensor(2.0)
y_true = torch.tensor(5.0) # 실제 정답

#순전파(Forward Propagation)
#모델의 예측값 계산
y_pred = w * x + b

loss = (y_pred - y_true) ** 2

print(f'예측값: {y_pred.item()}, 손실값: {loss.item()}')

#역전파(Backpropagation)
#손실에 대해 w와 b의 미분값 계산
loss.backward()

#w와 b의 미분값(기울기) 확인
#이 기울기를 이용해 w와 b를 업데이트하여 손실을 줄일 수 있음
print(f'w의 미분값: {w.grad}, b의 미분값: {b.grad}')

#예측값: 3.0, 손실값: 4.0
#w의 미분값: -8.0, b의 미분값: -4.0
```

* w와 b의 기울기가 계산된다. 이 기울기는 손실을 줄이기 위해 w와 b를 어느 방향으로 얼마나 조정해야 할지 알려준다. 예를 들어, w.grad가 양수라면, w를 감소시켜야 손실이 줄어든다는 의미(손실값을 줄이는 방향). PyTorch는 이 복잡한 과정을 loss.backward() 한 줄로 처리

* requires_true = True로 설정한 이유: 이 텐서들이 학습의 대상이기 때문. Loss를 줄이기 위해 w와 b를 조금씩 수정하는 과정이다. 이때 얼마나 수정할지 결정하려면 미분값이 필수인데, 이를 자동으로 구하기 위해 설정한다.

* $z = 2x^3$ x가 3일 때 z.backward()를 호출하면 x.grad는 54
