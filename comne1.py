# ================================
# 算法3: Fast-SCNN (Fast Segmentation Convolutional Neural Network)
# ================================
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class LearningToDownsample(nn.Module):
    """快速下采样模块"""

    def __init__(self):
        super(LearningToDownsample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dsconv1 = DepthwiseSeparableConv(32, 48, stride=2)
        self.dsconv2 = DepthwiseSeparableConv(48, 64, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class PyramidPoolingFastSCNN(nn.Module):
    """金字塔池化模块 - Fast-SCNN专用版本"""

    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingFastSCNN, self).__init__()
        self.pool_sizes = pool_sizes
        # 确保输出通道数是4的倍数，方便后续处理
        out_channels_per_branch = in_channels // 4

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels_per_branch, 1),
                nn.BatchNorm2d(out_channels_per_branch),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])

    def forward(self, x):
        h, w = x.shape[2:]
        out = [x]

        for conv in self.convs:
            pooled = conv(x)
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            out.append(pooled)

        return torch.cat(out, dim=1)


class GlobalFeatureExtractor(nn.Module):
    """全局特征提取器"""

    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()

        # Bottleneck blocks
        self.block1 = self._make_bottleneck(64, 64, 3, 1)
        self.block2 = self._make_bottleneck(64, 96, 3, 2)
        self.block3 = self._make_bottleneck(96, 128, 3, 1)

        # Pyramid pooling - 输出 128 + 32*4 = 256 通道
        self.ppm = PyramidPoolingFastSCNN(128, pool_sizes=[1, 2, 3, 6])

    def _make_bottleneck(self, in_channels, out_channels, repeats, stride):
        layers = []
        layers.append(DepthwiseSeparableConv(in_channels, out_channels, stride))
        for _ in range(repeats - 1):
            layers.append(DepthwiseSeparableConv(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.ppm(x)  # 输出256通道
        return x


class FeatureFusionModule(nn.Module):
    """特征融合模块"""

    def __init__(self, high_channels, low_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_high, x_low):
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x_high = F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear', align_corners=False)

        out = x_low + x_high
        return self.relu(out)


class Classifier(nn.Module):
    """分类器模块"""

    def __init__(self, in_channels, n_classes):
        super(Classifier, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels, 1)
        self.conv2 = DepthwiseSeparableConv(in_channels, in_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, n_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class FastSCNN(nn.Module):
    """Fast-SCNN - 超快速分割网络 (~1.1M参数)"""

    def __init__(self, n_classes=1):
        super(FastSCNN, self).__init__()

        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule(high_channels=256, low_channels=64, out_channels=128)
        self.classifier = Classifier(128, n_classes)

    def forward(self, x):
        input_size = x.size()[2:]

        # Learning to downsample: 1/8
        x_low = self.learning_to_downsample(x)

        # Global feature extractor: 1/32 -> 1/8 with PPM (输出256通道)
        x_high = self.global_feature_extractor(x_low)

        # Feature fusion
        x = self.feature_fusion(x_high, x_low)

        # Classifier
        x = self.classifier(x)

        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(x)