# CNN 모델 추가 학습

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
  plt.imshow(np.transpose(np.img, (1,2,0)))
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























