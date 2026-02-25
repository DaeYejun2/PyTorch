# Introducion
* 웨이퍼 제조 과정을 향상시키기 위한 다양한 웨이퍼 실패 패턴 감지 자동화
* 데이터 출처: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
* Model: ResNet-18
* output: 결함 패턴 라벨링

### 데이터 준비
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")

print("Path to dataset files:", path)
```

Path to dataset files: /root/.cache/kagglehub/datasets/qingyi/wm811k-wafer-map/versions/1


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = "/root/.cache/kagglehub/datasets/qingyi/wm811k-wafer-map/versions/1"
file_path = os.path.join(path, "LSWMD.pkl")

df = pd.read_pickle(file_path)

print(df.info())
```
<img width="520" height="370" alt="image" src="https://github.com/user-attachments/assets/2b0c0639-8e2e-462b-8c33-cc57dd7ccb04" />

```
df.head()
```
<img width="927" height="543" alt="image" src="https://github.com/user-attachments/assets/557efaa5-8d3d-40aa-abd0-a9fa997273c7" />

```
df.tail()
```
<img width="926" height="675" alt="image" src="https://github.com/user-attachments/assets/cb175d94-2d12-4b23-8021-3849285000f4" />

## 1. 데이터 필터링 및 기초 통계 확인

```
# model이 wafermap이랑 failuretype에만 집중할 수 있도록 불필요한 칼럼들(노이즈)은 삭제 해준다.
df.drop(['waferIndex', 'lotName', 'dieSize'], axis=1)

# failureType을 문자열/숫자로 변환하여 다루기 쉽게 만들기
# (원본은 리스트나 특수 객체 형태일 수 있음)
df['failureType'] = df['failureType'].astype(str)

# 3. 데이터 분포 확인
print("결함 패턴별 데이터 개수:")
print(df['failureType'].value_counts())
```

```
결함 패턴별 데이터 개수:
failureType
[]                 638507
[['none']]         147431
[['Edge-Ring']]      9680
[['Edge-Loc']]       5189
[['Center']]         4294
[['Loc']]            3593
[['Scratch']]        1193
[['Random']]          866
[['Donut']]           555
[['Near-full']]       149
Name: count, dtype: int64
```

none이나 []로 표시된 정상 데이터가 압도적으로 많다. 이 불균형을 처리해야 학습에 용이하다.

## 2. 데이터 정제 및 레이블 수치와
복잡한 문자열 형태를 깔끔하게 단어로 바꾸고, 학습에 방해 되는 빈 데이터([])를 처리해 준다.

ResNet-18 모델이 인식할 수 있도록 각 패턴을 숫자로 매핑하는 작업도 병행해준다.

```
# 문자열 정리 함수
def clean_label(x):
  x = x.replace('[', '').replace(']','').replace("'","")
  if x == "": return "unknown" # 비어있는 [] 처리
  return x

df['failureType'] = df['failureType'].apply(clean_label)

# unknown 데이터는 학습에서 제외 (패턴을 알 수 없기 때문)
df = df[df['failureType'] != 'unknown'].reset_index(drop=True)

# 레이블을 숫자로 변환
mapping_type = {
    'none': 0, 'Edge-Ring': 1, 'Edge-Loc': 2, 'Center': 3,
    'Loc': 4, 'Scratch': 5, 'Random': 6, 'Donut': 7, 'Near-full': 8
}
df['label'] = df['failureType'].map(mapping_type)

# 정제 후 데이터 분포
print(df['failureType'].value_counts())
```

```
failureType
none         147431
Edge-Ring      9680
Edge-Loc       5189
Center         4294
Loc            3593
Scratch        1193
Random          866
Donut           555
Near-full       149
Name: count, dtype: int64
```

## 3. 웨이퍼 맵 시각화
각 불량 패턴 출력

```
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12,12))
ax = ax.ravel() # 다차원 배열을 1차원 배열로

labels = ['none', 'Edge-Ring', 'Edge-Loc', 'Center', 'Loc', 'Scratch', 'Random', 'Donut', 'Near-full']

for i, label in enumerate(labels):
  # 해당 라벨의 첫 번째 행 추출
  subset = df[df['failureType'] == label].iloc[0]
  ax[i].imshow(subset['waferMap'], cmap='inferno')
  ax[i].set_title(f"{label} ({subset['waferMap'].shape})")
  ax[i].axis('off')

plt.tight_layout()
plt.show()
```
<img width="2296" height="2311" alt="image" src="https://github.com/user-attachments/assets/bfc65076-da15-45b5-8f28-5aeb5598d455" />

## 4. 이미지 전처리 및 채널 변환
웨이퍼의 크기가  45x48, 26x26 등으로 제각각이다. CNN 모델에 넣기 위해 모든 이미지를 동일한 크기로 맞추는 리사이징 작업이 필요하다.

딥러닝 모델(PyTorch)은 보통 (배치 크기, 채널, 높이, 너비) 형태의 텐서를 입력으로 받는다. 이번 단계에서는 다음을 수행한다.

1. 모든 웨이퍼 맵을 64x64 크기로 통일
2. 현재 0,1,2 등으로 구성된 값을 모델이 학습하기 좋은 형태로 정규화한다.
3. ResNt-18은 기본적으로 3채널(RGB) 입력을 기대하므로, 1채널인 웨이퍼 데이터를 3채널로 확장해준다.

```
import cv2

def resize_wafer(maps):
  # 모든 웨이퍼 맵 리사이징
  return cv2.resize(maps, dsize=(64,64), interpolation=cv2.INTER_NEAREST)

# 이미지 리사이징 적용
df['waferMap_resized'] = df['waferMap'].apply(resize_wafer)

# 모델 입력을 위한데이터 형태로 변환 (N, H, W)
X = np.stack(df['waferMap_resized'].values)
y = df['label'].values

# 채널 차원 추가 (N, H, W) -> (N, H, W, 1)
# ResNet을 위해 추후 (N, 3, H, W)로 바꿀 예정
X = X[:, :, :, np.newaxis]

print(f"전처리된 데이터 형태: {X.shape}") # (데이터 개수, 64,64,1)
print(f"전처리된 레이블 형태: {y.shape}")
```

```
전처리된 데이터 형태: (172950, 64, 64, 1)
전처리된 레이블 형태: (172950,)
```

INTER_NEAREST 옵션: 웨이퍼 맵의 값(0, 1, 2)은 범주형 데이터이다. 일반적인 보간법을 쓰면 1.5 같은 존재하지 않는 값이 생길 수 있어, 가장 가까운 이웃 값을 사용하는 방식을 써준다.

지금처럼 none 데이터가 상당히 많은 상태에서 모델을 돌리면, 모델은 학습을 안하고 무조건 정상(none)이라고 대답해도 정확도 90%를 얻게 된다.

이를 해결하기 위해 데이터 증강과 언더 샘플링을 사용하겠다.

## 5. 데이터 분할 및 불균형 해소
웨이퍼 데이터는 단순 이미지이므로, none 데이터는 개수를 줄이고 부족한 결함 데이터는 회전이나 반전을 통해 늘리는 것이 효과적이다. - > Undersampling 기법

```
from sklearn.model_selection import train_test_split

# 1. 학습과 테스트 데이터를 나눠준다. (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. 다시 학습 데이터를 학습과 검증으로 나눈다.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# 3. 데이터 불균형 처리를 위한 간단한 전략
# 'none' 데이터가 너무 많으므로 학습 데이터에서만 일부 추출하여 비중을 조절할 수 있다.
print(f"학습 데이터 형태: {X_train.shape}, 라벨 개수: {len(np.unique(y_train))}")
print(f"검증 데이터 형태: {X_val.shape}")
print(f"테스트 데이터 형태: {X_test.shape}")
```

```
학습 데이터 형태: (110688, 64, 64, 1), 라벨 개수: 9
검증 데이터 형태: (27672, 64, 64, 1)
테스트 데이터 형태: (34590, 64, 64, 1)
```

## 6. 특정 결함 패턴 데이터 증강
```
import torch
from torchvision import transforms

def augment_data(X_data, y_data):
  X_augmented = []
  y_augmented = []

  # 90도씩 회전 및 좌우 반전 조합
  for i in range(len(X_data)):
    img = X_data[i]
    label = y_data[i]

    # 정상(none)은 너무 많으므로 증강 제외
    if label == 0:
      X_augmented.append(img)
      y_augmented.append(label)
      continue

    # 결함 데이터는 8가지 변형 생성 (원본+3번 회전) x (좌우 반전 여부)
    for r in [0,1,2,3]: # 0,90,180,270 도
      rotated = np.rot90(img, k=r)
      X_augmented.append(rotated)
      y_augmented.append(label)

      flipped = np.fliplr(rotated) # 좌우 반전
      X_augmented.append(flipped)
      y_augmented.append(label)

  return np.array(X_augmented), np.array(y_augmented)

# 학습 데이터에만 증강 적용
X_train_aug, y_train_aug = augment_data(X_train, y_train)

print(f"증강 전 학습 데이터 수: {len(X_train)}")
print(f"증강 후 학습 데이터 수: {len(X_train_aug)}")
print("증강 후 라벨별 분포:")
unique, counts = np.unique(y_train_aug, return_counts=True)
print(dict(zip(unique, counts)))
```

```
증강 전 학습 데이터 수: 110688
증강 후 학습 데이터 수: 225012
증강 후 라벨별 분포:
{np.int64(0): np.int64(94356), np.int64(1): np.int64(49560), np.int64(2): np.int64(26568), np.int64(3): np.int64(21984), np.int64(4): np.int64(18400), np.int64(5): np.int64(6104), np.int64(6): np.int64(4440), np.int64(7): np.int64(2840), np.int64(8): np.int64(760)}
```

이 작업으로 none 대비 결함 데이터의 비중이 비약적으로 상승하여 모델이 결함 특징 학습을 용이하게 됐고, 모델이 특정 방향의 스크래치뿐만 아니라 회전된 형태의 스크래치도 동일한 결함으로 인식하게 된다.

## 7. ResNet-18 모델
```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

# 데이터를 PyTorch 텐서로 변환 (N,H,W,C) -> (N,C,H,W)
def to_tensor(X, y):
  # ResNet을 위해 채널을 3개로 복제 (1채널->3채널)
  X_tensor = torch.from_numpy(X).permute(0,3,1,2).float()
  X_tensor = X_tensor.repeat(1,3,1,1) # (N.3.64.64)
  y_tensor = torch.from_numpy(y).long()
  return TensorDataset(X_tensor, y_tensor)

train_dataset = to_tensor(X_train_aug, y_train_aug)
val_dataset = to_tensor(X_val, y_val)
test_dataset = to_tensor(X_test, y_test)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None) # 사전 학습된 가중치 없이 초기화

# 마지막 출력 레이어를 클래스 개수(9개)에 맞게 수정
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)

model = model.to(device)
```

model.fc 수정: 원래 ResNet-18은 1,000개의 클래스를 분류하도록 되어 있다. 하지만 위 데이터는 결함 종류가 9개 이므로 마지막 full Connected Layer를 9개로 바꿔주었다.

## 8. 손실 함수와 최적화 도구 설정 및 모델 학습
```
import torch.optim as optim

# 손실 함수 (다중 클래스 분류)
criterion = nn.CrossEntropyLoss()

# 2. 최적화 도구 (Adam: 학습률을 스스로 조절하며 빠르게 수렴)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
history = {'train_loss': [], 'val_acc': []}

for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0

  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad() # 기울기 초기화
    outputs = model(inputs) # 모델 예측
    loss = criterion(outputs, labels) # 오차 계산
    loss.backward() # 역전파
    optimizer.step() # 가중치 업데이트

    running_loss += loss.item() * inputs.size(0)

  epoch_loss = running_loss / len(train_loader.dataset)
  history['train_loss'].append(epoch_loss)

  # 평가
  model.eval()
  correct, total = 0, 0
  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  val_acc = 100 * correct / total
  history['val_acc'].append(val_acc)
  print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")

```

```
Epoch [1/10] Loss: 0.0274 | Val Acc: 97.32%
Epoch [2/10] Loss: 0.0222 | Val Acc: 97.40%
Epoch [3/10] Loss: 0.0190 | Val Acc: 97.43%
Epoch [4/10] Loss: 0.0170 | Val Acc: 97.42%
Epoch [5/10] Loss: 0.0156 | Val Acc: 97.46%
Epoch [6/10] Loss: 0.0137 | Val Acc: 97.36%
Epoch [7/10] Loss: 0.0135 | Val Acc: 97.34%
Epoch [8/10] Loss: 0.0125 | Val Acc: 97.02%
Epoch [9/10] Loss: 0.0114 | Val Acc: 97.24%
Epoch [10/10] Loss: 0.0109 | Val Acc: 97.33%
```

### 8-1 시각화
```
plt.figure(figsize=(12,5))

# Loss 그래프
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.title('Loss Trend')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(1,2,2)
plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
```
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/c7230135-a936-48f7-8800-16ea9819c46d" />

준수한 결과

## 9. 최종 평가 및 혼동 행렬 시각화
혼동 행렬(Confusion Matrix, 오차 행렬): 머신러닝 분류 모델의 성능을 평가하는 표로, 실제값과 모델ㅇ 예측한 값을 비교하여 2x2(또는 kxk) 형태로 시각화한 것이다.

```
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

target_names = ['none', 'Edge-Ring', 'Edge-Loc', 'Center', 'Loc', 'Scratch', 'Random', 'Donut', 'Near-full']

# 테스트 데이터 평가
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
  for inputs, batch_labels in test_dataset:
    # 개별 데이터를 배치 형태(1, 3, 64, 64)로 변환하여 예측
    inputs = inputs.unsqueeze(0).to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    y_true.append(batch_labels.item())
    y_pred.append(predicted.item())

# 혼동 행렬 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Wafer Fault Classification')
plt.show()

print(classification_report(y_true, y_pred, target_names=target_names))
```

<img width="806" height="701" alt="image" src="https://github.com/user-attachments/assets/1cddf88a-80ac-4a25-954e-bad447948343" />

```
              precision    recall  f1-score   support

        none       0.99      0.99      0.99     29486
   Edge-Ring       0.97      0.98      0.98      1936
    Edge-Loc       0.79      0.86      0.82      1038
      Center       0.91      0.92      0.91       859
         Loc       0.78      0.74      0.76       718
     Scratch       0.67      0.67      0.67       239
      Random       0.81      0.94      0.87       173
       Donut       0.87      0.90      0.88       111
   Near-full       0.83      1.00      0.91        30

    accuracy                           0.97     34590
   macro avg       0.85      0.89      0.87     34590
weighted avg       0.97      0.97      0.97     34590
```
* Accuracy: 97% 
* Macro Avg: 87%. 모든 클래스에 동일한 비중을 두고 계산한 평균이다. 개수가 적은 결함들(Scratch)의 낮은 점수가 반영되어 전체 정확도보다는 낮은 모습.
* Weighted Avg: 97%. 개수가 많은 클래스에 가중치를 둔 평균. none 데이터가 워낙 많아 정확도와 비슷하게 나온다.

### 정규화 된 지표
```
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12,10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Wafer Fault Classification')
plt.show()
```

<img width="924" height="855" alt="image" src="https://github.com/user-attachments/assets/de0e1048-8d46-49b8-a2bf-54a43c18fb90" />


## 10. Conclusion
정상 제품에서의 precision이 0.99로 상당히 좋은 성능을 보여준다. 공정 자동화 관점에서 정상 제품을 불량으로 오판해서 버리는 비용은 거의 없다는 뜻으로 볼 수 있겠다.

Edge-Ring이나 Near-full은 특징을 잘 파악해 패턴 인식이 잘 되었지만, Scratch나 Loc은 패턴이 뚜렷하지 않아서 인지 아쉬운 점수를 보였다.

ResNet-18을 활용하여 대규모 데이터셋의 불균형 문제를 해결하였으며, Macro F1-score 0.87을 달성함으로써 단순 정확도를 넘어 실제 결함 감지 역량을 입증하였다.
