# Deep Residual Learning for Image Recognition

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
* 과거에는 Vanishing Gradient(기울기 소실)이 문제였으나, 이는 BatchNormalization이나 Weight Initialization 기법으로 상당 부분 해결되었다.
* 네트워크가 어느 정도 깊어지면 정확도가 Saturate(포화)된 후 급격히 떨어지는 현상이 발견되었다.
* 이 현상은 Overfitting 때문이 아님을 알 수 있다.
<img width="504" height="179" alt="image" src="https://github.com/user-attachments/assets/28d76573-7e37-4a8c-8937-3757eb967d98" />
