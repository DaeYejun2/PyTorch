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








