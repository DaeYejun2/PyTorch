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
F1 Score: 0.08

|  | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.70 | 0.97 | 0.82 | 105 |
| 1 | 1.0 | 0.9523 | 0.97 | 0.97 |
| accuracy |  |  | 0.69 | 150 |
| macro avg | 0.55 | 0.51 | 0.45 | 150 |
| weighted avg | 0.61 | 0.69 | 0.60 | 150 |

* 수료자를 거의 맞추지 못하고 있는 상태

### F1 Score란
* 정밀도(Precision): 모델이 수료할거라고 예측한 사람 중 실제로 수료한 사람의 비율
* 재현율(Recall): 실제 수료자들 중 모델이 수료할거라고 골라낸 비율
* 정밀도와 재현율의 조화로운 평균이라고 보면 될것 같다.

## 1차 보강
```
# 1. 텍스트 길이로 수료율 예측
text_cols = ['whyBDA', 'what_to_gain', 'incumbents_lecture_scale_reason']
for col in text_cols:
  train_clean[f'{col}_len'] = train_clean[col].astype(str).apply(len)
  test_clean[f'{col}_len'] = test_clean[col].astype(str).apply(len)

# 2. 투입 시간
def clean_time(x):
  import re
  x = str(x)
  nums = re.findall(r'\d+', x) # 숫자만 추출
  if len(nums) == 2: # 3~4 형태
    return (float(nums[0]) + float(nums[1])) / 2
  elif len(nums) == 1:
    return float(nums[0])
  return 0.0

train_clean['time_input_num'] = train_clean['time_input'].apply(clean_time)
test_clean['time_input_num'] = test_clean['time_input'].apply(clean_time)

# 3. 이전 기수 수강 횟수
prev_cols = ['previous_class_3', 'previous_class_4', 'previous_class_5',
             'previous_class_6', 'previous_class_7', 'previous_class_8']

train_clean['participation_score'] = train_clean[prev_cols].notnull().sum(axis=1)
test_clean['participation_score'] = test_clean[prev_cols].notnull().sum(axis=1)


# 학습에 사용할 컬럼 선택
features = ['generation', 'school1', 'major_data', 'completed_semester',
            'time_input_num', 'participation_score', 'whyBDA_len', 'what_to_gain_len']

X_refined = train_clean[features]
y_refined = train_clean['completed']

# 검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_refined, y_refined, test_size=0.2, random_state=42, stratify=y_refined
)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 임계값 0.4 적용 후 점수
val_probs = model.predict_proba(X_val)[:, 1]
final_preds = (val_probs >= 0.4).astype(int)

print(f"F1 Score: {f1_score(y_val, final_preds):.4f}")

```
* F1 Score: 0.3261

## LightGBM 모델
LightGBM: 틀린 문제에 가중치를 두어 반복 학습한다. 특히 수료자(1)처럼 적은 데이터의 패턴을 파고드는 데 훨씬 강력하다.

```
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# 1. 사용할 컬럼 확장 (모델이 학습할 항목 늘리기)
# 기존 숫자 피처 + 의미 있을 법한 범주형 피처들
features = [
    'generation', 'school1', 'major_data', 'completed_semester',
    'time_input_num', 'participation_score', 'whyBDA_len', 'what_to_gain_len',
    'job', 'inflow_route', 'desired_career_path', 're_registration'
]

# 2. 추가된 범주형 피처들 Label Encoding
# train_clean과 test_clean에 적용
for col in ['job', 'inflow_route', 'desired_career_path', 're_registration']:
    le = LabelEncoder()
    # Train/Test 전체 범위를 맞추기 위해 합쳐서 fit
    full_data = pd.concat([train_clean[col], test_clean[col]], axis=0).astype(str)
    le.fit(full_data)
    train_clean[col] = le.transform(train_clean[col].astype(str))
    test_clean[col] = le.transform(test_clean[col].astype(str))

# 3. 데이터 준비
X_final = train_clean[features]
y_final = train_clean['completed']
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# 4. LightGBM 모델 생성 및 학습
# Random Forest보다 수료자(1)를 더 끈질기게 찾아낸다.
lgbm = lgb.LGBMClassifier(
    n_estimators=2000,       # 더 많이 학습하게
    learning_rate=0.01,      # 대신 더 천천히 (세심하게)
    max_depth=5,             # 너무 깊게 파서 과적합되는 것 방지
    num_leaves=31,
    subsample=0.8,           # 데이터의 80%만 무작위로 써서 학습 (일반화 성능 향상)
    colsample_bytree=0.8,    # 컬럼도 80%만 무작위 선택
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

lgbm.fit(X_train, y_train)

# 5. 성능 확인 및 최적 Threshold 찾기
lgbm_probs = lgbm.predict_proba(X_val)[:, 1]

print("--- LightGBM 결과 ---")
for t in [0.3, 0.35, 0.4, 0.45]:
    t_preds = (lgbm_probs >= t).astype(int)
    score = f1_score(y_val, t_preds)
    print(f"Threshold {t}: F1 Score = {score:.4f}")


# --- LightGBM 결과 ---
# Threshold 0.3: F1 Score = 0.3857
# Threshold 0.35: F1 Score = 0.4031
# Threshold 0.4: F1 Score = 0.3871
# Threshold 0.45: F1 Score = 0.4000
```

#### 키워드, 성실도 복합 지표 생성
```
passion_keywords = ['성장', '프로젝트', '취업', '데이터', '실무', '끝까지', '열심히', '공모전']

def count_passion(text):
    text = str(text)
    return sum(1 for word in passion_keywords if word in text)

for df in [train_clean, test_clean]:
    # 지원동기와 얻고싶은 것에서 키워드 개수 추출
    df['passion_score'] = df['whyBDA'].apply(count_passion) + df['what_to_gain'].apply(count_passion)

    최근 기수 가중치 지표
    # 8기 활동 여부에 가중치 (previous_class_8이 있으면 성실도 급상승)
    df['recent_activity'] = df['previous_class_8'].notnull().astype(int) * 2
    df['total_loyalty'] = df['participation_score'] + df['recent_activity']

# --- [데이터 재구성] ---
features = [
    'generation', 'school1', 'major_data', 'completed_semester',
    'time_input_num', 'total_loyalty', 'whyBDA_len', 'what_to_gain_len',
    'passion_score', 'job', 'inflow_route', 're_registration'
]

X = train_clean[features]
y = train_clean['completed']
test_X = test_clean[features]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- [필살기 3: 고성능 모델 세팅] ---
lgbm_pro = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.005, # 훨씬 더 세밀하게 학습
    max_depth=6,
    num_leaves=20,
    min_child_samples=10, # 데이터가 적으므로 과적합 방지
    class_weight='balanced',
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1
)

lgbm_pro.fit(X_train, y_train)

# --- [결과 확인] ---
probs = lgbm_pro.predict_proba(X_val)[:, 1]
best_f1 = 0
best_t = 0

for t in np.arange(0.2, 0.6, 0.05):
    preds = (probs >= t).astype(int)
    score = f1_score(y_val, preds)
    print(f"Threshold {t:.2f}: F1 Score = {score:.4f}")
    if score > best_f1:
        best_f1 = score
        best_t = t

print(f"\n최고 점수: {best_f1:.4f} (Threshold: {best_t:.2f})")
```

* 최고 점수: 0.4417 (Threshold: 0.20)

### 제출용 파일 저장
```
# 1. Test 데이터에 대해 확률값 예측
test_probs = lgbm_pro.predict_proba(test_X)[:, 1]

# 2. 최적의 문턱값(0.25) 적용
final_test_preds = (test_probs >= 0.25).astype(int)

# 3. 제출 파일 저장
submission['completed'] = final_test_preds
submission.to_csv('submission3.csv', index=False)
```

