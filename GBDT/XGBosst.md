## Before starting..
### AEP_MX 데이터
* AEP_MW는 미국 동부 지역의 대형 전력 회사인 American Electric Power의 MW(전력량 측정 단위)이다.
* 즉, 이 데이터는 AEP 전력 회사가 공급하는 지역의 사람들이 매시간 총 몇 MW의 전기를 썼는가를 기록한 로그이다.
### 목표
요일별/시간별 전력 수요 예측을 통해 효율적인 전력 생산을 가능케 한다.
<br>
### 참고
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
    n_estimators=1000,  # 최대 1000개의 트리
    learning_rate=0.01, # 오차를 줄여나가는 비율
    max_depth=5,        # decision tree의 깊이를 5까지
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
  * [0]번 단계: 처음에는 오차가 2578정도로 매우 크다. 학습한게 없는 상황
  * [700]번 단계: 오차가 885까지 떨어졌다. 
  * n_estimators를 1000으로 설정했지만, early_stopping_rounds=50 으로 과적합을 피했다.
    * 한번 최고 성적이 나오고 50회 동안 더 나아지지 않으면 조기 종료
   
# 3. 어떤 feature를 가장 중요하게 생각하는지
```
from xgboost import plot_importance

# 어떤 피처가 가장 중요했는지 그래프로 보기
plt.figure(figsize=(10, 8))
plot_importance(reg, height=0.9)
plt.show()
```
<img width="624" height="455" alt="image" src="https://github.com/user-attachments/assets/5b90c12f-d38a-4806-91df-589383d1bd87" />

1. dayofweek: 전력 소비 패턴이 평일과 주말에 따라 매우 극명하게 갈린다는 뜻.(공장이 돌아가는 날과 쉬는 날의 전력 사용 차이가 크다는 걸 캐치)
2. lag_24h: "어제 이 시간엔 에너지를 얼마나 썻지?" 라는 정보가 예측에 큰 도움을 준것으로 보인다.
3. lag_7d & hour: 일주일 전 데이터와 현재 시간대가 비슷한 중요도를 보인다.

# 실제 vs 예측 비교
CSV 파일에 들어있던 실제 데이터와 머신러닝으로 구한 예측값을 비교해보겠다.

```
# 1. 테스트 데이터에 대한 예측값 생성
test['prediction'] = reg.predict(X_test)

# 2. 시각화(최근 2주치(약 336시간))
plt.figure(figsize=(15, 6))
test['AEP_MW'][-336:].plot(label='Actual (Real)', color='blue', alpha=0.7)
test['prediction'][-336:].plot(label='Predicted (XGB)', color='red', linestyle='--')

plt.title('AEP Energy Consumption: Actual vs Predicted (Last 2 Weeks)')
plt.ylabel('Power Consumption (MW)')
plt.legend()
plt.grid(True)
plt.show()
```
<img width="1264" height="580" alt="image" src="https://github.com/user-attachments/assets/1d420de6-8e21-4757-acf2-0d69cae0b5d0" />

### 결과
* 높낮이가 바뀌는 구간(주말 추정)에서도 예측값과 실제값이 유사한 결과가 나오는 것을 확인할 수 있다.
* 가끔 Peak에서 모델이 실제보다 조금 높거나 낮게 예측하는 경우가 있는데, 날씨같은 외부 변수가 반영되지 않았기 때문일 가능성이 크다.
* 요일과 24시간 전 소비량이 전력 수요를 결정하는 가장 중요한 요소임을 데이터로 확인했고, 미국 동부 지역의 전력 소비가 주중/주말 패턴과 직전 시간대의 연속성에 매우 강하게 의존한다는 사실을 확인할 수 있다.


피처 엔지니어링(시간 추출, Lag 변수) -> 데이터 분할(시계열 방식) -> 모델 학습 및 검증 -> 시각화 과정을 수행했다.
