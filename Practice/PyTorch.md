## Tensor

---

숫자를 담는 다차원 배열. matrix를 일반화한 것

## PyTorch & Numpy

---
공통점: PyTorch와 Numpy 배열 모두 다차원 배열을 다룬다
차이점: 
1. PyTorch는 GPU 연산을 활용하여 연산 속도를 크게 높일 수 있다
대규모 데이터셋과 복잡한 모델을 다루는 딥러닝에서는 GPU활용이 필수적
2. 자동 미분
   PyTorch는 텐서에 대한 모든 연산의 미분값을 자동으로 계산해주는 Autograd 내장(경사 하강법 구현)
   
