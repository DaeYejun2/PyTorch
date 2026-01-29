## Before starting..
* n_estimators: 생성할 나무의 개수(너무 많으면 Overfitting)
* learnig_rate: 각 나무의 보정 강도(보통 0.01~0.1)
* max_depth: 나무의 깊이(보통 3~10, 센서 데이터처럼 복잡하면 깊게 조절)

# 1. 데이터 가져오기
*reference: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption*

<img width="1408" height="321" alt="image" src="https://github.com/user-attachments/assets/55c16dd9-eb8b-45d0-9239-bef75eed681b" />

```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('AEP_hourly.csv')
print(df.info())
print(df.head())
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
