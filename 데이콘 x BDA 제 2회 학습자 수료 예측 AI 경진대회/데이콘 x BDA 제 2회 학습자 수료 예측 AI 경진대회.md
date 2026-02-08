# Overview
* 주제: BDA 9기 데이터를 활용한 10기 학습자 수료 예측 (Binary Classification)
* 목표: 설문 정보를 바탕으로 중도 이탈 가능성이 높은 학습자를 식별하여 수료율 제고
* 평가: Binary F1 Score

# 구현

## 데이터 불러오기
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

print("Train Data Shape:", train.shape)
print("Test Data Shape:", test.shape)
print("-" * 30)
display(train.head())

# Train Data Shape: (748, 46)
# Test Data Shape: (814, 45)
# ------------------------------
```

```
print(train.info())

# 수료 여부 확인
target_col = 'completed'
print(train[target_col].value_counts())
```

## 데이터 전처리
```
# 데이터 전처리
# 1. 정보가 거의 없는 컬럼 삭제
drop_cols = ['contest_award', 'idea_contest', 'class3', 'class4', 'contest_participation']
train_clean = train.drop(columns=drop_cols)
test_clean = test.drop(columns=drop_cols)

# 2. 결측치 처리
# 2전공 없는 경우 None으로
train_clean['major1_2'] = train_clean['major1_2'].fillna('None')
test_clean['major1_2'] = test_clean['major1_2'].fillna('None')

# 3. 이전 기수 수강 횟수 feature 생성
prev_cols = ['previous_class_3', 'previous_class_4', 'previous_class_5',
             'previous_class_6', 'previous_class_7', 'previous_class_8']
train_clean['prev_count'] = train_clean[prev_cols].sum(axis=1)
test_clean['prev_count'] = test_clean[prev_cols].sum(axis=1)

print("전처리 후 컬럼 수: ", train_clean.shape[1])

# 전처리 후 컬럼 수:  42
```

## Random Forest 모델
```
# object 타입을 숫자로 바꾼 뒤 모델 학습
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# 1. 불필요한 ID 및 데이터가 없는 칼럼 제거
X = train_clean.drop(columns=['ID', 'completed']) # ID는 예측에 사용하지 않으므로 제외
y = train_clean['completed']
test_X = test_clean.drop(columns=['ID'])

# 2. Label Encoding(모든 문자열을 숫자로 변환)
# Train과 Test에 동일한 기준을 적용하기 위해 합쳐서 인코딩 하거나
# 결측치를 미리 문자열 'NAN'으로 채워야 에러가 나지 않는다.
combined = pd.concat([X, test_X], axis=0)

# object나 bool 타입의 컬럼들만 골라서 변환
cols_to_encode = combined.select_dtypes(include=['object', 'bool']).columns

for col in cols_to_encode:
    le = LabelEncoder()
    combined[col] = combined[col].astype(str).fillna('Unknown')
    combined[col] = le.fit_transform(combined[col])

# 다시 Train / Test로 분리
X_encoded = combined.iloc[:len(X), :]
test_X_encoded = combined.iloc[len(X):, :]

# 3. 검증 데이터 분할 (수료자 비율 유지를 위해 stratify=y 사용)
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 4. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 성능 검증 확인
val_preds = model.predict(X_val)
print("F1 Score:", f1_score(y_val, val_preds))
print("\nClassification Report:\n", classification_report(y_val, val_preds))
```

#### stratify=y
현재 데이터는 수료자가 적은 불균형 데이터이다. 그냥 나누게 되면 상황에 따라 운 좋게 검증 데이터에 수료자가 너무 적게 들어갈 수 있기 때문에,
stratify=y를 쓰면 학습용과 검증용 데이터셋에 수료(1)와 미수료(0)의 비율을 원본과 똑같이(약 3:7) 맞춰서 나눠준다. 

### 결과
|  | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.70 | 0.97 | 0.82 | 105 |
| 1 | 1.0 | 0.9523 | 0.97 | 0.97 |
| accuracy |  |  | 0.69 | 150 |
| macro avg | 0.55 | 0.51 | 0.45 | 150 |
| weighted avg | 0.61 | 0.69 | 0.60 | 150 |








