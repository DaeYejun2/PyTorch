## 1. Explainability
불량 요소를 볼 때 웨이퍼의 어느 영역을 보고 그렇게 판단했는지 시각화하는 것이 필수적이다.

Grad-CAM을 통해 노이즈가 아닌 실제 결함 패턴에 집중하고 있음을 증명할 수 있다. 예를 들어, Edge-Loc 판정 시 실제로 가장자리 부분의 활성도가 높게 나타나는지 확인한다.

Grad-CAM은 모델의 마지막 컨볼루션 레이어에서 나오는 Feature Map과 그라디언트를 결합하여, **모델이 이미지의 어느 부분을 보고 이 클래스로 예측했는지** 히트맵으로 보여주는 기술.

이 웨이퍼는 Scratch입니다. -> 이 부근의 픽셀 흐름이 Scratch의 전형적인 패턴과 일치하여 89%의 확률로 판단했다.

### Grad-CAM 구현 코드
```
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 그래디언트와 활성화 맵을 추출하기 위한 후크(Hook) 등록
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        # 1. 순전파
        self.model.eval()
        output = self.model(input_tensor)
        
        # 2. 역전파 (특정 클래스에 대해)
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # 3. 그래디언트 평균 계산 (Global Average Pooling)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. 활성화 맵과 가중치 결합
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # 5. ReLU 적용 (양수 영향력만 추출) 및 정규화
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()

# 사용 예시: ResNet-18의 마지막 컨볼루션 층인 'layer4' 지정
target_layer = model.layer4[-1] 
grad_cam = GradCAM(model, target_layer)
```

### 시각화
```
def visualize_gradcam(idx, dataset, target_names):
    input_tensor, label = dataset[idx]
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 모델 예측
    output = model(input_batch)
    _, pred = torch.max(output, 1)
    
    # 히트맵 생성
    heatmap = grad_cam.generate_heatmap(input_batch, pred.item())
    
    # 시각화
    img = input_tensor[0].cpu().numpy() # 3채널 중 첫 번째 채널 사용
    heatmap_resized = cv2.resize(heatmap, (64, 64))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(f"Actual: {target_names[label]}\nPredicted: {target_names[pred.item()]}")
    ax[0].axis('off')

    # 히트맵을 원본 이미지 위에 투명하게 겹침
    ax[1].imshow(img, cmap='gray')
    ax[1].imshow(heatmap_resized, cmap='jet', alpha=0.5) # alpha로 투명도 조절
    ax[1].set_title("Grad-CAM (Focus Area)")
    ax[1].axis('off')
    
    plt.show()

# label이 5번(Scratch)인 데이터들의 인덱스 찾기  0부터 8까지
for i in range(9):
  indices = np.where(y_test == i)[0]
  print(f"Scratch 데이터 인덱스들: {indices[:10]}") # 앞에서 10개만 출력

  visualize_gradcam(indices[0], test_dataset, target_names)
# 테스트 데이터 중 결함 데이터(예: Scratch) 인덱스 하나 골라서 실행
```

<img width="598" height="1575" alt="image" src="https://github.com/user-attachments/assets/20a741fc-18b1-4045-9773-dfa1925a2f76" />

<img width="596" height="1286" alt="image" src="https://github.com/user-attachments/assets/80e2763b-3860-4652-8bcc-ba593b910f05" />

Random, Conut, Near-full과 같이 잘 맞춘 패턴들의 공통점은 웨이퍼의 특정 '면적'이나 '범위'를 차지하는 Global한 특징을 가지고 있다. 이로 인해 현재의 모델은 전반적인 픽셀의 밀도와 위치를 파악하는데는 좋은 성능을 보인다.

반면 Scratch는 면적이 아닌 가느다란 선이다. 현재 모델은 이미지의 Shape보다는 분포에 의존해 학습된 상태이다.

현재까지의 결과로 확인할 수 있는 결론은, Grad-CAM은 면적 기반 결함에 대해서는 결함 위치를 정확히 특정하며 높은 신뢰도를 보인다. 하지만 선형 결함의 경우, 모델이 결함의 국소적인 특징보다 거시적인 픽셀 밀도에 집중하는 경향이 있어 검출률이 저하됨을 확인하였다.

이를 해결하기 위해 모델이 스크래치의 '선'을 물리적으로 식별할 수 있도록 해상도를 높여주는 방식으로 데이터를 준비시켜보겠다.

기존 64x64 에서 128x128로 해상도를 변경해서 확인해보겠다.

<img width="637" height="430" alt="image" src="https://github.com/user-attachments/assets/7245c094-60c6-4914-9b45-bc64a48cf868" />

해상도 상향의 Precision이 0.67에서 0.79로 유의미한 변화가 있긴 하지만, 여전히 모델이 스크래치를 정상과 구별하기엔 특징이 너무 약하다고 느끼는 것 같다.

어쩔 수 없이 Scratch 전용 가중치 부여를 해보겠다. 작동 방식은 Scratch를 none이라고 예측했을 때 받는 Loss가 너무 작기 때문이다. Loss를 더 높여 더 민감한 반응을 유도한다.

<img width="614" height="424" alt="image" src="https://github.com/user-attachments/assets/a59ba9c2-46e5-4316-8423-4716741842ba" />

가중치를 높이니 모델이 조금이라도 결함이 느껴지면 스크래치라고 반응하게 되어 진짜 스크래치는 더 많이 찾았지만(Recall 증가), 멀쩡한 노이즈를 스크래치(Precision)로 오해하기 시작한 것이다.

128x128 해상도에서도 여전히 선의 특징이 점(노이즈)의 특징과 유사하게 판단하는 것 같다.

ResNet-18모델은 아주 미세한 픽셀 단위의 연결성을 파악하기보다는 전체적인 패턴에 더 강점이 있는 것 같다.

