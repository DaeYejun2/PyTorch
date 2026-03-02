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

<img width="794" height="415" alt="image" src="https://github.com/user-attachments/assets/2cd58ab0-c10d-438a-b108-199c67fac17c" />

