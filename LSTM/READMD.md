Dataset reference: https://archive.ics.uci.edu/dataset/179/secom
<br>
This dataset contains 1,567 examples of semiconductor manufacturing process data with 590 features from various sensors.

## Why use LSTM not RNN
일반적인 RNN은 시계열 데이터를 다룰 수 있지만, 치명적인 한계가 있다.
1. 기울기 소실 문제
  * RNN은 문장이나 시계열 데이터가 길어지면, 앞부분의 정보가 뒤로 전달되지 못하고 사라지는데 이를 **기울기 소실**이라 한다.
  * LSTM은 Cell State라는 별도의 공간을 만들어, 중요한 정보는 수많은 스템이 지나도 손실 없이 끝까지 전달한다.
2. 선별적 기억 능력
<br>
LSTM은 3가지 Gate를 통해 정보를 관리한다.

* 쓸모없는 과거 정보(일시적 노이즈 같은)를 삭제
* 현재 들어온 정보 중 중요한 것만 기억
* 가공된 기억 중 어떤 부분을 최종 출력으로 내보낼지 결정
