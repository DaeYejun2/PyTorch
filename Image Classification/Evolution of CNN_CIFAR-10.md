# CNN 모델의 진화

## CNN 모델 기본 코드
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
```
<img width="543" height="176" alt="image" src="https://github.com/user-attachments/assets/8dad6249-fed8-4140-957b-a6340304b054" />

```
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn. Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16*5*5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
  running_loss = 0.0

  for i, data in enumerate(train_loader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 2000 == 1999:
      print("Epoch: {} | Batch: {} | Loss: {}".format(epoch+1, i+1, running_loss/2000))
      running_loss = 0.0
```

Epoch: 1 | Batch: 2000 | Loss: 2.16815944981575 .....
<br>
Epoch: 5 | Batch: 12000 | Loss: 1.0179827889576554

```
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(' '.join('\t{}'.format(classes[labels[j]]) for j in range(4)))

net = Net().to(device)
net.load_state_dict(torch.load(PATH))

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)
print(' '.join('\t{}'.format(classes[predicted[j]]) for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
  for data in test_loader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  print(100 * correct / total)
```

cat 	ship 	plane 	plane
<br>
60.18

```
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
  for data in test_loader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
      label = labels[i]
      class_correct[label] += c[i].item()
      class_total[label] += 1
for i in range(10):
  print("Accuracy of {}: {}%".format(classes[i], 100*class_correct[i] / class_total[i]))
```
Accuracy of plane: 72.9%
<br>
Accuracy of car: 82.7%
<br>
Accuracy of bird: 51.7%
<br>
Accuracy of cat: 33.0%
<br>
Accuracy of deer: 61.1%
<br>
Accuracy of dog: 34.4%
<br>
Accuracy of frog: 69.2%
<br>
Accuracy of horse: 65.7%
<br>
Accuracy of ship: 78.0%
<br>
Accuracy of truck: 53.1%

```
def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(img, (1,2,0)))
  plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(12, 4))
for idx in range(4):
  ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
  img = images[idx] / 2 + 0.5
  plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

  color = "green" if predicted[idx] == labels[idx] else "red"
  ax.set_title(f"Target: {classes[labels[idx]]}\n(pred: {classes[predicted[idx]]})", color=color)

plt.tight_layout()
plt.show()
```
<img width="1189" height="344" alt="image" src="https://github.com/user-attachments/assets/9ad73b01-432f-4349-b109-042a58174953" />

## 문제점
* 어느 정도 학습은 되었지만, 배를 비행기로 착각하는 등 세밀한 특징을 잡는 데는 부족함이 보인다.
* 정확도를 올릴 수 있는 방법을 시도해보겠다.

## 1. Epoch 5->10
* Accuracy: 60.18% -> 62.7%
###### Accuracy of plane: 67.5%
###### Accuracy of car: 84.9%
###### Accuracy of bird: 41.1%
###### Accuracy of cat: 46.5%
###### Accuracy of deer: 54.0%
###### Accuracy of dog: 55.7%
###### Accuracy of frog: 68.4%
###### Accuracy of horse: 71.4%
###### Accuracy of ship: 68.8%
###### Accuracy of truck: 68.7%
* 유의미한 증가가 있긴 하지만, 정체된 몇몇 항목들이 보인다.

## 2. 데이터 증강 적용
```
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
```
* Accuracy: 62.58%
###### Accuracy of plane: 62.6%
###### Accuracy of car: 77.7%
###### Accuracy of bird: 52.6%
###### Accuracy of cat: 45.8%
###### Accuracy of deer: 62.0%
###### Accuracy of dog: 50.3%
###### Accuracy of frog: 71.8%
###### Accuracy of horse: 62.8%
###### Accuracy of ship: 72.2%
###### Accuracy of truck: 68.0%

현재의 작은 모델 구조(LetNet-5)로는 학습 시간을 늘리거나 데이터 증강을 적용하는 것만으로는 더 이상 정확도가 오르지 않는 한계치에 도달했음을 확인할 수 있다.

## 모델 용량 키우기
```
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(64 * 8 * 8, 512)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 64*8*8)
    x = self.dropout(F.relu(self.fc1(x)))
    x = self.fc2(x)
    return x

net = Net().to(device)
```
* 채널 수 확장 (6/16 -> 32/64)
  * Convolution 필터의 개수를 대폭 늘려 모델의 기억 용량을 키우는 작업이다.
* 배치 정규화
  * 학습 과정에서 각 층을 통과할 때마다 데이터의 분포를 일정하게 맞춰주는 역할을 한다.
  * 기울기 소실 문제를 줄여 주어 학습 속도가 향상되고, 초기 학습률 설정에 덜 예민해져 안정성을 갖춘다.
* Dropout
  * 학습 중에 무작위로 일부 뉴럭을 0으로 만드는(꺼버리는) 기법
    * Overfitting 방지: 특정 뉴런 몇 개에만 의존해서 정답을 외우는 것을 방지한다.
* padding 추가
  * 이미지 가장자리에 가상의 픽셀(보통 0)을 둘러준다.
  * Convolution 연산을 하면 이미지 크기가 조금씩 줄어드는데, 패딩을 쓰면 입력과 출력 크기를 유지할 수 있어 가장자리 부분의 정보도 충분히 활용할 수 있게 된다.

<br>

* Accuracy: 72.68
###### Accuracy of plane: 78.6%
###### Accuracy of car: 81.5%
###### Accuracy of bird: 68.4%
###### Accuracy of cat: 52.3%
###### Accuracy of deer: 60.6%
###### Accuracy of dog: 64.8%
###### Accuracy of frog: 75.3%
###### Accuracy of horse: 85.3%
###### Accuracy of ship: 80.1%
###### Accuracy of truck: 83.1%

* 유의미한 정확도 변화를 보여준다.
* 위와 같은 구조는 현대 딥러닝의 전환점이 된 VGGNet(VGG-16)의 기술들과 닮아 있다.
  1. 작은 필터(3x3)와 Padding의 조합
     * 기존 LetNet-5는 5x5 필터를 써서 이미지를 빠르게 훑었지만, VGGNet은 3x3 필터를 여러 번 사용하는 방식을 사용한다.
  2. 단계적인 채널 수 확장
     * VGGNet은 층이 깊어질수록 채널을 32->64->128->256 식으로 두 배씩 늘려가는 구조를 표준화한다.
  3. 완전 연결 계층(Fc Layer)의 비중 축소
     * 과거 모델들은 마지막 분류 단계에서 엄청나게 큰 파라미터를 사용했지만, 현대 모델들은 fc의 크기를 적절히 조절하고 Dropout을 배치하여 **과도한 파라미터로 인한 과적합**을 방지한다.
    
## Transfer Learning
끝판왕 전이학습
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

import torchvision.models as models

# ResNet18 모델 불러오기
net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
  running_loss = 0.0

  for i, data in enumerate(train_loader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 2000 == 1999:
      print("Epoch: {} | Batch: {} | Loss: {}".format(epoch+1, i+1, running_loss/2000))
      running_loss = 0.0

PATH = './resnet18_cifar.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(' '.join('\t{}'.format(classes[labels[j]]) for j in range(4)))

net = net.to(device)
net.load_state_dict(torch.load(PATH))

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)
print(' '.join('\t{}'.format(classes[predicted[j]]) for j in range(4)))

net.eval()
correct = 0
total = 0

with torch.no_grad():
  for data in test_loader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  print(100 * correct / total)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
  for data in test_loader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
      label = labels[i]
      class_correct[label] += c[i].item()
      class_total[label] += 1
for i in range(10):
  print("Accuracy of {}: {}%".format(classes[i], 100*class_correct[i] / class_total[i]))

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(img, (1,2,0)))
  plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(12, 4))
for idx in range(4):
  ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
  img = images[idx] / 2 + 0.5
  plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

  color = "green" if predicted[idx] == labels[idx] else "red"
  ax.set_title(f"Target: {classes[labels[idx]]}\n(pred: {classes[predicted[idx]]})", color=color)

plt.tight_layout()
plt.show()
```

* Accuracy: 92.1
###### Accuracy of plane: 90.3%
###### Accuracy of car: 97.4%
###### Accuracy of bird: 93.8%
###### Accuracy of cat: 74.7%
###### Accuracy of deer: 92.0%
###### Accuracy of dog: 88.3%
###### Accuracy of frog: 96.9%
###### Accuracy of horse: 97.4%
###### Accuracy of ship: 94.3%
###### Accuracy of truck: 95.9%

압도적인 정확도.
<br>

* CIFAR-10으로 가져온 이미지는 32x32 크기의 공식 규격을 가지고 있기 때문에 Resize(224)를 해줘야 한다.
* net.eval(): 기존의 모델은 구조가 단순했기 때문에 괜찮았지만, ResNet18처럼 복잡한 현대적인 모델에서는 배치 정규화 층으로 인해 학습할 때와 테스트할 때 동작 방식이 완전히 달라진다.

전이학습 성능이 상당하다.

## 시각화
```
def visualize_model(model, num_images=6):
    # 모델의 현재 모드(train/eval)를 기억해둡니다.
    was_training = model.training
    # 평가 모드로 전환 (BatchNorm, Dropout 작동 방식 고정)
    model.eval()
    
    images_so_far = 0
    # 전체 그림 사이즈 설정
    plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                # num_images 개수만큼 격자를 만듭니다.
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                
                # 예측값과 실제 정답을 제목에 표시
                color = "green" if preds[j] == labels[j] else "red"
                ax.set_title(f'predicted: {classes[preds[j]]}\n(actual: {classes[labels[j]]})', color=color)
                
                # 이미지 역정규화 및 출력
                img = inputs.cpu().data[j] / 2 + 0.5
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))

                if images_so_far == num_images:
                    # 함수 종료 전 원래 모델 모드로 복구
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.show()
                    return
        
        model.train(mode=was_training)
        plt.tight_layout()
        plt.show()

# 함수 호출
visualize_model(net, num_images=6)
```
<img width="695" height="789" alt="image" src="https://github.com/user-attachments/assets/3db02ba5-0872-4bc9-9b6a-2cc605d6ea25" />

plane의 특성이 bird와 유사해서 그런지 정확도가 높아도 오류가 나타남을 확인할 수 있다.

