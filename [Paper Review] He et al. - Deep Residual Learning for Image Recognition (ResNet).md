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
<br>
*Figure 1. Training error(left) and test error(right) on CIFAR-10 with 20-layer and 56-layer "plain" network*
<br>
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
<br>
*Figure2. Residual learning: a building block*
<br>
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
* H(x)
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


