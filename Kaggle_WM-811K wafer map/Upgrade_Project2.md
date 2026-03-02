## High-Resolution Network
해상도를 줄이지 않고 끝까지 유지하는 모델이다.

이전 프로젝트에서 끝까지 잡지 못했던 Scratch의 미세한 특징을 훨씬 잘 잡을 것이다.

일반적인 ResNet은 이미지를 직렬로 통과시키며 크기를 계속 줄인다. $224 \to 112 \to 56 \to 28 \to 14 \to 7$. 이 과정에서 1픽셀짜리 스크래치 정보는 뒤로 갈수록 작아진다.

반면, HRNet은 이름 그대로 고해상도를 버리지 않는다.

* 병렬 구조: 고해상도 줄기와 저해상도 줄기를 동시에 운영
* 지속적인 융합: 저해상도에서 얻은 전체적인 문맥과 고해상도에서 유지한 정보를 계속 주고받는다.


## HRNet
```
import torch
import torch.nn as nn
import torch.nn.functional as F

# 0. 하이퍼파라미터 및 설정
BN_MOMENTUM = 0.1

# 1. Bottleneck (부품 1)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None: identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)

# 2. BasicBlock (부품 2)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None: identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# 3. HighResolutionModule (부품 3)
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1: return None
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i: fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        num_outchannels_conv3x3 = num_inchannels[i] if k == i - j - 1 else num_inchannels[j]
                        conv3x3s.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                            nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1: return [self.branches[0](x[0])]
        for i in range(self.num_branches): x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + (x[j] if i == j else self.fuse_layers[i][j](x[j]))
            x_fuse.append(self.relu(y))
        return x_fuse

    def get_num_inchannels(self):
        return self.num_inchannels
```

```
class HighResolutionNet(nn.Module):
    def __init__(self, num_classes=9): # 1. 클래스 개수를 9로 설정
        super(HighResolutionNet, self).__init__()
        # 2. 입력을 3채널(RGB)로 변경 (기존 1에서 3으로 수정)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        stage1_out_channel = Bottleneck.expansion * 64

        # Stage 2
        num_channels = [18, 36]
        block = BasicBlock
        num_channels = [nc * block.expansion for nc in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(2, block, [4, 4], num_channels)

        # Stage 3
        num_channels = [18, 36, 72]
        block = BasicBlock
        num_channels = [nc * block.expansion for nc in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(3, block, [4, 4, 4], num_channels)

        # Stage 4
        num_channels = [18, 36, 72, 144]
        block = BasicBlock
        num_channels = [nc * block.expansion for nc in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(4, block, [4, 4, 4, 4], num_channels)

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)

        # 3. 마지막 분류 레이어를 9개 클래스로 수정 (기존 10에서 9로 수정)
        self.classifier = nn.Linear(2048, num_classes)

    # --- 아래 _make_ 메서드들은 기존 코드와 동일하므로 그대로 둡니다 ---
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        transition_layers = []
        for i in range(len(num_channels_cur_layer)):
            if i < len(num_channels_pre_layer):
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - len(num_channels_pre_layer)):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - len(num_channels_pre_layer) else in_channels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        layers.extend(block(inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _make_stage(self, num_branches, block, num_blocks, num_channels, multi_scale_output=True):
        num_modules = num_branches
        num_inchannels = num_channels
        modules = nn.ModuleList()
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output if i < num_modules - 1 else False
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, 'SUM', reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        incre_modules = nn.ModuleList([self._make_layer(head_block, ch, hc, 1, stride=1)
                                       for ch, hc in zip(pre_stage_channels, head_channels)])
        downsamp_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hc * head_block.expansion, hc_next * head_block.expansion, 3, 2, 1, bias=False),
                nn.BatchNorm2d(hc_next * head_block.expansion, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ) for hc, hc_next in zip(head_channels[:-1], head_channels[1:])
        ])
        final_layer = nn.Sequential(
            nn.Conv2d(head_channels[-1] * head_block.expansion, 2048, 1, 1, 0),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x for i in range(2)]
        y_list = self.stage2(x_list)

        x_list = [self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i] for i in range(3)]
        y_list = self.stage3(x_list)

        x_list = [self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i] for i in range(4)]
        y_list = self.stage4(x_list)

        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        return y
```

```
model = HighResolutionNet(num_classes=9).to(device)
```
실행 결과는

