# Deep Residual Learning for Image Recognition
##### Reference: https://arxiv.org/abs/1512.03385

## abstract
* Deeper neural networks are more difficult to train.
* 네트워크의 깊이가 깊어질수록 학습이 어려워지는 문제를 해결하기 위해 Residual Learning 프레임워크를 제안한다.
* 레이어를 입력값에 대한 Residual function을 학습하도록 재구성하여, 깊은 네트워크에서도 최적화가 용이하도록 설계했다.
* 기존 VGG 네트워크보다 8배 깊은 최대 152층 까지 층을 쌓았음에도 불구하고, 오히려 복잡도는 더 낮게 유지했다.
* 깊이가 상당히 증가했음에도 불구하고 오차율을 낮추면서 정확도를 얻을 수 있음을 증명했다.

## 1. Introduction
### 1.1 The importance of Depth
* 딥러닝에서 네트워크의 **깊이**는 매우 중요하다. 층을 깊게 쌓을수록 낮음/중간/높은 수준의 특징들을 통합하여 더 풍부한 표현을 학습할 수 있기 때문이다.
* 실제로 ImageNet 대회에서 우승한 모델들은 16층(VGG)에서 30층에 이르는 매우 깊은 모델들을 사용해 왔다.
### 1.2 The Bottleneck: Degradation Problem (성능 저하 문제)
그렇다고 단순히 층을 많이 쌓는다고 해서 항상 성능이 좋은 것도 아니다.
* 과거에는 Vanishing Gradient(기울기 소실)이 문제였으나, 이는 BatchNormalization이나 Weight Initialization(가중치 초기화.Relu 같은) 기법으로 상당 부분 해결되었다.
* BatchNormalization: 배치 정규화. 네트워크가 깊어지면 이전 층의 가중치가 바뀔 때마다 다음 층으로 들어오는 값의 분포가 계속 변하는데, 이 때문에 학습 속도가 느려지고 불안정해지는 것을 막기 위해 학습 과정에서 각 층을 통과할 때마다 데이터의 분포를 강제로 일정하게 만들어주는 기법.
* 네트워크가 어느 정도 깊어지면 정확도가 Saturate(포화)된 후 급격히 떨어지는 현상이 발견되었다.
* 이 현상은 Overfitting 때문이 아님을 알 수 있다.
<img width="504" height="179" alt="image" src="https://github.com/user-attachments/assets/28d76573-7e37-4a8c-8937-3757eb967d98" />

*Figure 1. Training error(left) and test error(right) on CIFAR-10 with 20-layer and 56-layer "plain" network*

* 56층 모델이 20층 모델보다 Training Error 자체가 더 높게 나타난다. 즉, 모델이 학습 자체를 못하고 있다는 증거이다.

### 1.3 The Hypothesis
* 얕은 모델A와 여기에 층을 더 쌓은 깊은 모델 B가 있을 때, 깊은모델B가 최소한 얕은 모델A만큼의 성능을 내려면, 추가된 층들이 Identity Mapping(입력을 그대로 출력으로 보냄)만 수행하면 된다.
* 이론적으로는 깊은 모델이 더 낮은 오차를 가져야 하지만, 실제 실험 결과 기존의 방식으로는 단순한 Identity Mapping조차 학습하기 어려워한다는 것을 발견했다

### 1.4 Solution: Deep Residual Learning
#### 1. Residual Learning
* 기존처럼 레이어가 목표 매핑 (H(x))을 직접 학습하게 하는 대신, Residual Mapping을 학습하도록 구조를 바꿨다.
* 잔차 함수인 F(x) = H(x) - x를 학습한다. 결과 적으로 원래의 목표 F(x) + x를 구하는 것이 된다.
* 이렇게 한면 만약 Identity Mapping이 최적이라면, 레이어들은 단순히 가중치를 0으로 수렴 시키기만 하면 되므로 학습이 훨씬 쉬워진다.
* x: 레이어에 들어가기 전의 원본 데이터
* F(x): x가 여러 개의 레이어(Convolution, Batch Norm 등)를 통화하며 계산된 결과물
* H(x): 최종 답

#### 2. Shortcut Connection
* 이러한 F(x) + x 구조는 Shortcut Connection을 통해 구현된다.
<img width="320" height="168" alt="image" src="https://github.com/user-attachments/assets/229559e0-a4d3-48a6-83cb-897fce295f3b" />

*Figure2. Residual learning: a building block*

* 하나 이상의 레이어를 건너뛰고 입력값(x)을 레이어의 출력값에 직접 더해준다.
* 추가적인 파라미터나 계산 복잡도를 늘리지 않는다는 장점이 있다.
* 기존읜 경사하강법(SGD)과 오차 역전파를 그대로 사용할 수 있으며, Caffe같은 라이브러리에서도 별도의 수정 없이 쉽게 구현 가능하다.

#### 3. Verification Results
* ImageNet과 CIFAR-10(데이터 셋) 데이터를 이용한 실험
* 매우 깊은 Residual Nets는 최적화가 쉬운 반면, 단순히 층을 쌓은 Plain nets는 깊어질 수록 학습 오차가 높아졌다
* 층이 깊어질 수록 정확도가 눈에 띄게 좋아졌으며, 이전의 다른 네트워크들보다 뛰어난 성능을 보였다
*CIFAR-10에서는 100층을 넘어 1,000층 이상의 모델까지 학습에 성공했다.*

* 152층의 ResNet은 당시 ImageNet에서 가장 깊은 네트워크였지만, VGG 네트워크보다 연산량(복잡도)은 더 낮았다.
* 이 모델로 구성된 앙상블은 ILSVRC 2015 분류 부문에서 3.57%의 오차율로 1위를 차지했다.
* 층은 152층으로 훨씬 깊어졌지만, 실제 연산량이나 파라미터의 복잡도는 VGG보다 더 낮거나 효율적이라는 사실을 통해, Residual Learning 구조를 쓰면 훨씬 효율적으로 고성능 모델을 만들 수 있음을 보여준다.

## 2. Related Work
### 2.1 Residual Representations
* VLAD & Fisher Vector: 이미지 검색이나 분류에서 쓰이는 이 기법들은 dictionary와의 차이를 인코딩하는 방식이다. 원래 벡터를 그대로 쓰는 것보다 residual(잔차)를 쓰는 것이 더 효과적이라는 점이 이미 증명되어 있다.
* Multigrid Method: 편미분 방정식(PDE)을 풀 때 사용하는 이 방법은 문제를 여러 스케일로 나누어 풀며, 각 하위 문제는 거친 스케일과 세밀한 스케일 사이의 residual solution을 담당한다.
* 결론: 이러한 사례들은 문제를 residual형태로 재구성 하는 것이 최적화를 훨씬 단순하게 만들 수 있음을 시사한다.

### 2.2 Shortcut Connections
* 레이어를 건너뛰는 shortcut 구조 역시 완전히 새로운 것은 아니지만, ResNet만의 차별점이 존재한다.
* Early Practice: 초기 다층 퍼셉트론(MLP) 연구에서도 입력층과 출력층을 집접 연결하는 선형 레이어를 추가하곤 했다.
* GoogLeNet(Inception): 인셉션 구조에도 shortcut branch가 포함되어 있다.
* Highway Networks와의 차이: ResNet과 가장 유사한 동시대 연구는 'Highway Networks'이지만, 결정적인 차이가 있다.
  * Highway Networks: 데이터에 따라 지름길을 열고 닫는 **Gating**함수가 있고 파라미터가 필요하다. 게이트가 닫히면 residual을 학습하지 않게 된다.
  * ResNet: 파라미터가 전혀 없는 Identity Shortcut을 사용하여 항상 정보가 흐르게 하여, 항상 residual함수만을 학습한다. 또한 Highway Networks는 층이 아주 깊어질 때(100층 이상)의 성능 향상을 증명하지 못했다.


## 3. Deep Residual Learning
### 3.1 Residual Learning
H(x)를 네트워크가 해결해야할 최적의 매핑이라고 정의할 때, 
<br>
* 핵심 가설: 여러 비선형 레이어가 복잡한 함수 H(x)를 직접 학습하는 것보다, residual 함수인 $F(x) := H(x) - x$를 학습하는 것이 훨씬 쉽다.
* 만약 Identity 매핑이 최적이라면, 레이어들은 단순히 가중치를 0으로 수렴시켜 F(x) = 0을 만들면 된다. 이는 아무것도 없는 상태에서 H(x) = x가 되도록 정교하게 가중치를 맞누는 것보다 훨씬 단순한 최적화 문제이다.

### 3.2 Identity Mapping by Shortcut
이 가설을 실제 구조로 만든 것이 바로 Residual Block이다.
<br>
**$$y = F(x, \{W_i\}) + x$$**
* x: 레이어의 입력
* $$F(x, \{W_i\})$$: 학습해야 할 Residual 매핑
* 입력 x를 레이어의 끝단에서 단순히 더해준다.
* 파라미터가 늘어나지 않고, 연산량 증가도 거의 없으며, 역전파 시 기울기가 x를 타고 그대로 앞단까지 전달된다.
* 입력 x와 레이어의 출력 F(x)의 차원이 다를 때(예: Stride를 사용해 크기가 줄어든 경우)는
  * Zero-padding: 부족한 차원을 0으로 채워서 더함(파라미터 증가 없음).
  * Projection Shortcut($W_s$): 1x1 컨볼루션을 사용하여 차원을 강제로 맞춘다
    * **$y = F(x, \{W_i\}) + W_sx$**
  ```
  def forward(self, x):
  identity = x # 입력 x를 따로 저장

  out = self.conv1(x)
  out = self.bn1(out)
  out = self.relu(out)

  out = self.conv2(out)
  out = self.bn2(out)

  # F(x) + x
  out += identity

  out = self.relu(out)

  return out
  ```
* identity에 x를 따로 저장해서 나중에 더해주지 않는다면, 레이어들은 입력 x를 완전히 새로운 H(x)로 통째로 바꾸는 어려운 작업을 해야한다.

### 3.3 Network Architectures
* Plain Network: VGG의 철학을 따라 단순히 3x3 컨볼루션을 쌓은 모델(지름길 없음)
  * 층이 깊어질 수록 정답을 찾는 과정이 너무 복잡해져 학습이 제대로 안되는 성능저하 문제 발생
* Residual Network: 위 Plain 구조에 2개의 레이어마다 Shortcut을 추가한 모델
  * 2개의 레이어를 건너뛰는 이유는 레이어가 최소한의 의미있는 잔차를 만들어 낼 수 있는 공간을 확보해주기 위함.
  * 레이어가 정답(H(x))을 통째로 만드는 대신, 입력(x)과 정답의 차이인 F(x)만 만들게 해, F(x) = H(x) - x라는 식을 만들어 레이어의 임무를 정답을 만드는 것에서 부족한 차이를 메우는 것으로 재정의한 것이다.
<img width="494" height="1128" alt="image" src="https://github.com/user-attachments/assets/c9d56c69-b91f-4376-8600-21d613e877eb" />

*Example network architectures for ImageNet. Left: the VGG-19 model Middle: a plain network with 34 parameter layers. Right: a residual network with 34 parameter layers*

### 3.4 Implementation
1. Data Augmentation
   * Scale Augmentation: 이미지의 짧은 쪽 길이를 256~480 사이에서 랜덤하게 조절
   * Cropping & Flipping: 위 이미지에서 $224 \times 224$ 크기를 랜덤하게 잘라내거나 좌우 반전을 수행
   * Color Augmentation: 표준적인 색상 변형 기법을 적용
   * Normalization: 픽셀별 평균값을 빼주는 전처리를 수행
  
2. Training Strategy
   * Batch Normalization (BN): 모든 합성곱(Conv) 직후, 그리고 활성화 함수(ReLU) 직전에 적용
   * Optimizer: SGD(Stochastic Gradient Descent)를 사용하며, Momentum은 0.9, Weight Decay(L2 규제)는 0.0001로 설정
   * Learning Rate (LR): 0.1에서 시작하여, 에러가 더 줄어들지 않고 정체(plateau)되면 10으로 나누어(0.01, 0.001...) 줄여나감
   * No Dropout: Batch Normalization을 사용하기 때문에 드롭아웃은 사용하지 않는다.
   * Iterations: 최대 $60 \times 10^4$번의 반복 학습을 수행하며 미니배치 크기는 256
  
3. Testing
   * 10-crop testing: 하나의 이미지를 중앙, 모서리 등 10가지 방식으로 잘라 테스트한 뒤 평균을 낸다.
   * Multi-scale testing: 이미지를 5가지 크기({224, 256, 384, 480, 640})로 리사이징하여 각각 점수를 매긴 뒤 평균을 내어 정확도를 극대화한다.

## 4. Experiments
*1,000개의 클래스를 가진 ImageNet 2012 데이터셋으로 성능을 검증*
### Plain Networks
```
import torch
import torch.nn as nn

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()
        # 3x3 convolution 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 3x3 convolution 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 지름길(Shortcut) 없이 바로 ReLU 통과
        out = self.relu(out)
        return out
```

### Residual Networks
```
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2(out_channels)

    # 차원이 변할 때(stride > 1) 입력을 맞춰주기 위한 shortcut
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2(in_channels, out_channels, Kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += self.shortcut(identity)

    out - self.relu(out)
    return out
```
```
self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2(in_channels, out_channels, Kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels)
```
* 차원 맞추기 로직
* 입력값 x를 출력값에 더해주려면 두 값의 크기(가로, 세로, 채널 수)가 반드시 같아야 한다. 그런데 층을 지나다 보면 이 크기가 변할 때가 있는데, 그 때 사용하는 코드
* ResNet은 모든 블록에서 똑같은 크기를 유지하지 않는다. 성능을 위해 중간중간 다음과 같은 변화를 준다
* 이미지 크기 축소: stride=2를 써서 가로,세로를 절반으로 줄인다
* 채널 수 증가: 더 깊은 정보를 담기 위해 채널을 64에서 128로 늘린다.
* self.shortcut = nn.Sequential(): 기본적으로 아무것도 안하는 코드. 크기가 같을 때는 그냥 통과하면되기 때문
* if stride != 1 or in_channels != out_channels: : 크기가 줄어들었거나 채널수가 변했다
* nn.Conv2d(..., kernel_size=1, stride=stride, ...): 1x1 Convolution을 사용하여 x의 크기를 조절한다
* stride=stride: out이 절반으로 줄었다면, x도 똑같이 절반으로 줄여준다
* out_channels: out의 채널이 늘어났다면 x의 채널도 똑같이 늘려준다.
<br>
<img width="718" height="325" alt="image" src="https://github.com/user-attachments/assets/1b0893be-9eec-4e4d-bc49-5bb5d227d490" />
<br>
모델의 깊이에 따른 세부 레이어 구성표. 본 프로젝트는 이 표를 바탕으로 18층과 34층 모델을 구현.
<br>

<img width="909" height="288" alt="image" src="https://github.com/user-attachments/assets/7ccd141a-44a1-4163-a691-ec96e78028b7" />

*Figure 4. Training on ImageNet*
<br>
Plain 모델에서는 깊이가 깊어질 수록 오차가 커지는 문제가 발생하지만, ResNet은 Shortcut path를 통해 이를 해결했다.
<br>

<img width="316" height="93" alt="image" src="https://github.com/user-attachments/assets/b198787c-9332-4fce-8686-c299ae1ddfac" />

<br>

*error (%, 10-crop testing) on ImageNet validation.*
<br>
<img width="346" height="296" alt="image" src="https://github.com/user-attachments/assets/e1ced55e-0fcc-4494-9608-861013d49395" />


* Table 3. Error rates (%, 10-crop testing) on ImageNet validation. VGG-16 is based on our test. ResNet-50/101/152 are of option B that only uses projections for increasing dimensions.
* Option A: 차원이 증가할 때만 0을 채움
* Option B: 차원이 증가할 때만 1x1 Conv 사용(본 프로젝트에서 구현한 방식)
* Option C: 모든 지름길에 1x1 Conv 사용
* B가 A보다 우수하며, C는 B와 비슷하지만 연산 효율성을 위해 B를 표준으로 채택한다.
#### CIFAR-10 데이터를 사용해 Plain-34와 ResNet-34 모델을 학습시키고 성능 비교
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 모델 블록 정의 (Plain vs Residual)
class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, is_plain=False):
    super(Block, self).__init__()
    self.is_plain = is_plain
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

    self.shortcut = nn.Sequential()
    if not is_plain or (stride != 1 or in_channels != out_channels):
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels)
      )
  
  def forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if not self.is_plain:
      out += self.shortcut(identity)

    return out

# 2. ResNet/Plain 모델 구조 정의
class ResNetOrPlain(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10, is_plain=False):
    super(ResNetOrPlain, self).__init__()
    self.in_channels = 64
    self.is_plain = is_plain
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    # Table 1의 34-layer 구성: [3, 4, 6, 3]
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride):
    stride = [stride] + [1]*(num_blocks-1)
    layers = []
    for s in stride:
      layers.append(block(self.in_channels, out_channels, s, self.is_plain))
      self.in_channels = out_channels
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
    x = self.avgpool(x)
    return self.fc(torch.flatten(x, 1))
```

```
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

def train_and_get_history(model, train_loader, epochs=10):
  model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

  history = []
  model.train()
  
  for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    history.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

  return history

print("--- Training Plain-34 ---")
plain_34 = ResNetOrPlain(Block, [3, 4, 6, 3], is_plain=True) #
plain_history = train_and_get_history(plain_34, train_loader, epochs=10)

print("\n--- Training ResNet-34 ---")
resnet_34 = ResNetOrPlain(Block, [3, 4, 6, 3], is_plain=False) #
resnet_history = train_and_get_history(resnet_34, train_loader, epochs=10)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), plain_history, label='Plain-34', color='red', marker='o')
plt.plot(range(1, 11), resnet_history, label='ResNet-34', color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss: Plain vs ResNet (CIFAR-10)')
plt.legend()
plt.grid(True)
plt.show()
```
<img width="307" height="653" alt="image" src="https://github.com/user-attachments/assets/4105b19d-ac6c-4b16-a50a-690398cc7a79" />
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/4055cc81-bac4-416b-ab99-05825e1a5ee2" />

*왜 초기 10Epoch에서 ResNet의 Loss가 더 높게 나타났는가?*
* Figure 4 그래프를 보면, 학습 극초반에는 Plain과 ResNet의 오차 곡선이 교차하거나 Plain이 일시적으로 낮게 유지되는 구간이 존재한다.
* ResNet의 진정한 성능은 학습이 더 진행되어 Plain 모델의 오차가 정체되는 시점에서, Shortcut을 통해 기울기 소실을 방지하며 오차를 끝까지 낮출 때 증명된다.

### 4.2 CIFAR-10 and Analysis
<img width="804" height="215" alt="image" src="https://github.com/user-attachments/assets/27cf68d2-0e4a-4091-8a84-3fb675227dbc" />

*Figure 6. Training on CIFAR-10. Dashed lines denote training error, and bold lines denote testing error. Left: plain networks. The error of plain-110 is higher than 60% and not displayed. Middle: ResNets. Right: ResNets with 110 and 1202 layers.*

1. 실험 설정
   * ImageNet과는 다른 CIFAR-10만의 특화된 설정을 사용
   * 입력 크기: 32x32 이미지 사용
   * 레이어 구성: 처음에 3x3 Conv 한 번, 그 뒤에 2n개의 레이어를 가진 세 가지 스택(이미지 크기 32, 16, 8)을 쌓아 총 6n+2개의 레이어로 구성
   * Shortcut: 모든 지름길은 Identity Shortcut을 사용(Option A)
2. 분석 1: 극단적인 깊이에서의 성능
   * Plain Nets의 한계: 층이 깊어질 수록(20 -> 32 -> 44 -> 56) 에러율이 점점 높아지는 성능 저하 문제가 명확히 드러났다.
   * 반면 ResNet은 층이 깊어질 수록 에러율이 낮아 졌으며, 110층 모델에서도 성능 향상이 지속되었다.
3. 분석 2: 1,000층 이상
   * 학습은 성공적이었고(오차가 낮게 유지), 110층 모델과 유사한 Training Error를 보였다.
   * 하지만 Test Error는 110층 모델(6.43%)보다 1,202층 모델(7.93%)이 더 높게 나왔다.
   * CIFAR-10 규모에 비해 모델이 너무 커서 발생한 Overfitting 때문이며, 이는 Regularization이 추가로 필요함을 시사한다.
<img width="383" height="112" alt="image" src="https://github.com/user-attachments/assets/918779b7-feea-4f4f-9bd1-3c2e94c1df0b" />

<img width="372" height="88" alt="image" src="https://github.com/user-attachments/assets/88ccbaff-7611-4a49-998e-d854e1881f15" />

4. 사물 탐지로의 확장
   * PASCAL VOC: 기존 VGG-16 기반보다 약 3% 이상 높은 성능을 보인다
   * MS COCO: 사물 탐지 성능이 VGG-16 대비 상대적으로 28% 향상되었다.
   * ResNet으로 배운 특징이 매우 강력하며, 어떤 비전 작업에 가져다 써도 성능이 오는다는 것을 의미한다.

## 5. Conclusion
* 딥러닝에서 **깊이**는 성능의 핵심이지만, 이를 가능케 하는 것은 단순한 적층이 아니라 최적화 가능한 구조(Residual Learning)임을 알 수 있다.
