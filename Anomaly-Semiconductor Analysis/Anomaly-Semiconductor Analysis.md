# 1. 데이터 읽기
scikit-learn에서 제공하는 라이브러리를 통해 실제 반도체 공정(SECOM)데이터를 가져온다.
```
from sklearn.datasets import fetch_openml
import pandas as pd

# 1. SECOM 데이터셋 불러오기 (OpenML ID: 410)
# 'as_frame=True'를 설정하면 바로 Pandas 데이터프레임으로 가져온다
secom = fetch_openml(data_id=410, as_frame=True, parser='auto')

# 2. 데이터 구성 확인
X = secom.data    # 590개의 센서 특징들 (온도, 압력 등 공정 변수)
y = secom.target  # 합격/불량 라벨

# 데이터 합치기
df = pd.concat([X, y], axis=1)

print(f"데이터 크기: {df.shape}")
df.head()
```
데이터 크기: (37, 1143)
<br>

<img width="983" height="284" alt="image" src="https://github.com/user-attachments/assets/f0bb4822-3fa6-4f61-9782-7d4e8ef3c8b9" />

```
import matplotlib.pyplot as plt
import seaborn as sns

# 상위 4개 센서 데이터의 분포 확인
plt.figure(figsize=(15, 5))
for i in range(1, 5):
  plt.subplot(1, 4, i)
  sns.histplot(df[f'oz{i}'], kde=True)
  plt.title(f'Sensor oz{i} Distribution')
plt.tight_layout()
plt.show()
```
<img width="1490" height="490" alt="image" src="https://github.com/user-attachments/assets/42f3aa01-0903-48e6-9dfd-f0d1d4f57d3e" />

* 스케일 확인: x축이 0~1 사이임을 확인하여, 이미 정규화가 되어 있음을 파악(안되어 있으면 추가 스케일링)
* 이상치 탐색: 데이터가 몰려있는 곳 외에 툭 튀어나온 값들이 있는지 확인하여, 이상탐지 모델이 필요하다는 근거를 찾음
* 공정 특성 파악: Mulit-modal을 확인하여, 반도체 공정에 여러 레시피가 섞여 있음을 짐작할 수 있다.

# 2. 데이터 전처리
반도체 공정 데이터는 센서 오작동이나 통신 오류로 인해 값이 비어있거나(NaN), 분석에 방해가 되는 가짜 데이터가 많다. 이를 해결해보자.
```
# 결측치가 50% 이상인 컬럼 삭제
df_cleaned = df.dropna(thresh=len(df)*0.5, axis=1)

# 남은 결측치는 중앙값으로 채우기
# 평균값은 이상치에 민감하지만, 중앙값은 더 안정적이다
df_cleaned = df_cleaned.fillna(df_cleaned.median())



























