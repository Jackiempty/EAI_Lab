import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Bottleneck Block
# ----------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, out_channels=None, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        # 若沒有提供 out_channels，就依照原始 ResNet 結構自動推導
        if out_channels is None:
            out_channels = [planes, planes, planes * self.expansion]
        else:
            assert len(out_channels) == 3, "Bottleneck requires out_channels list of length 3"
        # assert len(out_channels) == 3, "Bottleneck requires out_channels list of length 3"
        
        ################################################
        # Please replace ??? with the correct variable #            
        # example: in_channels, out_channels[0], ...   #
        ################################################
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        # self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.conv3 = nn.Conv2d(out_channels[1], planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        # downsample path (若 stride ≠ 1 或 channel 不匹配)
        # if downsample is None and (stride != 1 or in_channels != out_channels[2]):
        #     # print("not match channel, in:{}, out:{}".format(in_channels, out_channels[2]))
        #     self.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels[2], kernel_size=1, stride=stride, bias=False),
        #         # nn.BatchNorm2d(out_channels[2]),
        #     )
        # else:
        #     self.downsample = downsample
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # print("before downsample", identity.shape)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            # print("the downsample: ", self.downsample)
            # print("downsample conv weight:", self.downsample[0].weight.shape)
            # print("conv3 weight:", self.conv3.weight.shape)
            # print("bn3 weight:", self.bn3.weight.shape)
            identity = self.downsample(identity)
            # print("after downsample", identity.shape)

        out += identity
        out = self.relu(out)
        return out

# ----------------------
# ResNet
# ----------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, cfg, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.current_cfg_idx = 0

        # Conv1
        self.conv1 = nn.Conv2d(in_channels, cfg[self.current_cfg_idx], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[self.current_cfg_idx])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.current_cfg_idx += 1
        self.inplanes = cfg[self.current_cfg_idx]

        # Layer1~Layer4
        self.layer1 = self._make_layer(block, 64, layers[0], cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.inplanes, num_classes)

    
    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        #############################################################################
        # Figure out how to generate the correct layers and downsample based on cfg #
        #############################################################################
        downsample = None

        first_block_cfg = cfg[self.current_cfg_idx : self.current_cfg_idx + 3]
        in_channels = self.inplanes
        out_channels = first_block_cfg[2]

        # if stride != 1, do downsample
        if stride != 1 or in_channels != out_channels:
            out_channels = planes * 4
            # print("down sample in resnet, in:{}, out:{}".format(in_channels, out_channels))
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                # nn.BatchNorm2d(out_channels),
            )

        layers = []
        # first block（might need downsample）
        # print("parameters: [in_channels: {}, planes: {}, cfg: {}]".format(in_channels, planes, first_block_cfg))
        layers.append(
            block(in_channels, planes, first_block_cfg, downsample=downsample, stride=stride)
        )

        # update channels and cfg index
        self.inplanes = out_channels
        self.current_cfg_idx += 3

        # rest of blocks（stride=1, downsample=None）
        for i in range(1, blocks):
            block_cfg = cfg[self.current_cfg_idx : self.current_cfg_idx + 3]
            layers.append(
                block(self.inplanes, planes, block_cfg, downsample=None, stride=1)
            )
            self.inplanes = block_cfg[2]
            self.current_cfg_idx += 3

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*6 + \
              [512, 512, 2048]*3
    layers = [3, 4, 6, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet101(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*23 + \
              [512, 512, 2048]*3
    layers = [3, 4, 23, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet152(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*8 + \
              [256, 256, 1024]*36 + \
              [512, 512, 2048]*3
    layers = [3, 8, 36, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)