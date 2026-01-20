##  퍼셉트론과 MLP 

### before starting
인공 신경망(Artificial Neural Network)의 가장 기본이 되는 두 가지 구조, 퍼셉트론(Perceptron)과 다층 퍼셉트론(MLP)에 대해 알아볼 것이다. 

### 1. 퍼셉트론: 신경망의 조상
* 작동 원리
  1. 입력 신호와 가중치 곱하기: 여러 개의 입력 신호(x_1, x_2,...)와 각각의 가중치(w_1, w_2,...)를 곱한다. 가중치는 각 입력 신호가 결과에 미치는 중요도를 나타낸다.
  2. 모든 신호 더하기: 곱셈 결과를 모두 더한다. 여기에 **편향(Bias)** b를 더한다. 편향은 뉴런이 얼마나 쉽게 활성화 되는지를 조절한다.
  3. 활성화 함수 통과: 합산된 값이 특정 임계치(Threshold)를 넘으면 1을 출력하고, 넘지 못하면 0을 출력한다. 이 임계치를 넘는 기준을 결정하는 것이 활성화 함수(Activation Function)이다.
퍼셉트론은 간단한 논리 회로(AND, OR)를 구현할 수 있지만, XOR 문제와 같이 복잡한 비선형 문제(직선으로 분리할 수 없는 문제)는 해결할 수 없다는 한계가 있다.(딥러닝 침체기 원인 중 하나..)

### 2. 다층 퍼셉트론(MLP)

* MLP 구조
  1. 입력층(Input Layer): 외부로부터 데이터를 입력받는 층
  2. 은닉층(Hidden Layer): 입력층과 출력층 사이에 위치한 층으로, 여러 개를 쌓아 올릴 수 있다. 이 층에서 복잡한 패턴과 특징을 학습한다. 딥러닝이 여기서 유래했다.
  3. 출력층(Output Layer): 최종 결과를 출력하는 층
MLP는 은닉층을 통해 비선형 문제를 해결할 수 있게 되었다. 각 뉴련이 선형 변환(wx+b)을 수행한 후 비선형 활성화 함수를 통과하면서 복잡한 데이터의 특징을 잡아낼 수 있게 되었다.
활성화 함수로는 Sigmoid, ReLU 등이 주로 사용된다.

### 3. MLP 구현
```
import torch
import torch.nn as nn

#MLP 모델 클래스 정의
class SimpleMLP(nn.Module):
  def __init__(self):
    super().__init__()
    # 첫 번째 은닉층: 10개의 입력 뉴런, 32개의 뉴런
    self.hidden1 = nn.Linear(in_features=10, out_features=32)
    # 두 번째 은닉층: 32개의 입력 뉴런, 16개의 출력 뉴런
    self.hidden2 = nn.Linear(in_features=32, out_features=16)
    # 출력층: 16개의 입력 뉴런, 1개의 출력 뉴런
    self.output = nn.Linear(in_features=16, out_features=1)
    # 활성화 함수: ReLU(렐루)
    self.relu = nn.ReLU()

  def forward(self, x):
    # 순전파
    x = self.relu(self.hidden1(x))
    x = self.relu(self.hidden2(x))
    x = self.output(x)
    return x

# 모델 객체 생성
model = SimpleMLP()
print(model)

# 임의의 입력 데이터 생성 (1개 샘플, 10개 특성)
random_input = torch.randn(1, 10)
# 모델에 입력 데이터 통과
output = model(random_input)
print(f'\n입력 데이터 크기: {random_input.shape}')
print(f'모델 출력 크기: {output.shape}')

# SimpleMLP(
#   (hidden1): Linear(in_features=10, out_features=32, bias=True)
#   (hidden2): Linear(in_features=32, out_features=16, bias=True)
#   (output): Linear(in_features=16, out_features=1, bias=True)
#   (relu): ReLU()
# )

# 입력 데이터 크기: torch.Size([1, 10])
# 모델 출력 크기: torch.Size([1, 1])
```
위 코드에서는 입력층 10개, 은닉층 32개, 16개, 출력층 1개로 구성된 MLP를 정의했다. nn.Linear가 자동으로 가중치와 편향을 관리해주기 때문에, 사용자는 층의 구조와 활성화 함수만 정의하면 된다.

#### ReLU
* 데이터에 비선형성을 더해준다.
* 직선(선형) 연산만 계속 반복하면 아무리 층을 많이 쌓아도 결국 하나의 큰 직선이 될 뿐이지만, ReLU는 복잡하게 꺾인 곡선 형태의 데이터 패턴도 학습할 수 있게 만들어준다.
* 입력값이 0보다 작으면 0으로 만들고, 0보다 크면 그대로 통과시킨다. 이 과정에서 수식이 '꺾이게' 되는데, 이렇게 직선을 꺾어주는 힘이 바로 비선형.
