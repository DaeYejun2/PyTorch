## Before starting..
* n_estimators: 생성할 나무의 개수(너무 많으면 Overfitting)
* learnig_rate: 각 나무의 보정 강도(보통 0.01~0.1)
* max_depth: 나무의 깊이(보통 3~10, 센서 데이터처럼 복잡하면 깊게 조절)

# 1. 데이터 가져오기
*reference: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption*

<img width="1408" height="321" alt="image" src="https://github.com/user-attachments/assets/55c16dd9-eb8b-45d0-9239-bef75eed681b" />

<br>

```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('AEP_hourly.csv')
print(df.info())
print(df.head())

# -------------- 출력값 -----------------
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 121273 entries, 0 to 121272
# Data columns (total 2 columns):
#  #   Column    Non-Null Count   Dtype  
# ---  ------    --------------   -----  
#  0   Datetime  121273 non-null  object 
#  1   AEP_MW    121273 non-null  float64
# dtypes: float64(1), object(1)
# memory usage: 1.9+ MB
# None
#               Datetime   AEP_MW
# 0  2004-12-31 01:00:00  13478.0
# 1  2004-12-31 02:00:00  12865.0
# 2  2004-12-31 03:00:00  12577.0
# 3  2004-12-31 04:00:00  12517.0
# 4  2004-12-31 05:00:00  12670.0
```

<br>

```
# 3. Datetime 칼럼을 실제 시간 데이터 타입으로 변환
df['Datetime'] = pd.to_datetime(df['Datetime'])

# 4. 시간순으로 정렬(시계열 모델의 필수 단계)
df = df.sort_values('Datetime')

# 5. Datetime을 인덱스로 설정
df = df.set_index('Datetime')

# 6. 간단한 시각화로 데이터 흐름 확인
df.plot(figsize=(15, 5), title='AEP Hourly Energy Consumption')
plt.show()
```

<img width="1236" height="441" alt="image" src="https://github.com/user-attachments/assets/6f4b536f-8405-43d4-bd15-649df28e5dc0" />

## 1-1. XGBoost용 features 만들기
```
def create_features(df):
  df = df.copy()
  df['hour'] = df.index.hour
  df['dayofweek'] = df.index.dayofweek
  df['quarter'] = df.index.quarter
  df['month'] = df.index.month
  df['year'] = df.index.year
  df['dayofyear'] = df.index.dayofyear

  # 주말 여부 추가 (월=0, ... , 일=6)
  df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

  # 1. 과거 데이터 피처 생성
  # shift(n)은 데이터를 n칸 뒤로 미는 함수
  # 24시간 전, 7일 전 데이터를 추가
  df['lag_24h'] = df['AEP_MW'].shift(24)   # 어제 동일 시간 소비량
  df['lag_7d'] = df['AEP_MW'].shift(24*7)  # 지난주 동일 요일/시간 소비량

  return df

df = create_features(df)

print(df[['AEP_MW', 'hour', 'dayofweek', 'is_weekend']].head())
#                       AEP_MW  hour  dayofweek  is_weekend
# Datetime                                                 
# 2004-10-15 01:00:00  12766.0     1          4           0
# 2004-10-15 02:00:00  12159.0     2          4           0
# 2004-10-15 03:00:00  11972.0     3          4           0
# 2004-10-15 04:00:00  11867.0     4          4           0
# 2004-10-15 05:00:00  12021.0     5          4           0
```

# 2. 모델 학습을 위한 데이터 분할(Train/Test Split)
현재 상태와 과거 패턴을 동시에 고려할 수 있게 만들어준다.

```
# shift를 하면 처음 며칠간은 과거 데이터가 없어서 NaN이 생긴다
# 학습을 위해 결측치가 있는 행은 제거
df = df.dropna()

# 2. 없데이트된 피처 리스트
# 기존 시간 정보에 '과거 값' 두 개를 추가
FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 
            'is_weekend', 'lag_24h', 'lag_7d']
TARGET = 'AEP_MW'

# 3. 데이터 분할(2017년 기준)
train = df.loc[df.index < '2017-01-01']
test = df.loc[df.index >= '2017-01-01']

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# XGBoost 모델 학습
import xgboost as xgb
reg = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    early_stopping_rounds=50,
    random_state=42
)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
```

<img width="889" height="612" alt="image" src="https://github.com/user-attachments/assets/b99959b7-f28f-4506-aec3-345852d9a4a4" />

* 출력된 로그에서 가장 중요한 건 rmse
  * RMSE(Root Mean Squared Error): 예측값과 실제값의 차이(오차)를 나타낸다. 숫자가 작을수록 정확한 모델이라는 뜻
  * validation_0: 학습한 모델의 점수
  * validation_1: 모델의 테스트 점수
* 로그 점수
  * [0]번 단계: 처음에는 오차가 2580정도로 매우 크다. 학습한게 없는 상황
  * [800]번 단계: 오차가 958까지 떨어졌다. 
  * n_estimators를 1000으로 설정했지만












