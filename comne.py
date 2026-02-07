#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海岸线检测算法对比: 5种高效遥感分割算法
SegNet vs PSPNet vs Fast-SCNN vs LinkNet vs ENet
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import json
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CoastalDataset(Dataset):
    """海岸线数据集"""

    def __init__(self, image_paths, label_paths, transform=None, image_size=(512, 512)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.load_image(self.image_paths[idx])
        mask = self.create_mask_from_labelme(self.label_paths[idx], image.size)

        image = image.resize(self.image_size, Image.LANCZOS)
        mask = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)
        mask = np.array(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask

    def load_image(self, image_path):
        try:
            return Image.open(image_path).convert('RGB')
        except:
            return Image.new('RGB', (512, 512), (128, 128, 128))

    def create_mask_from_labelme(self, label_path, image_size):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            mask_image = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(mask_image)

            for shape in label_data.get('shapes', []):
                if shape['label'].lower() in ['water', 'sea', '海水', '水体']:
                    points = [(int(p[0]), int(p[1])) for p in shape['points']]
                    if len(points) >= 3:
                        draw.polygon(points, fill=1)

            return np.array(mask_image, dtype=np.uint8)
        except:
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)


# ================================
# 算法1: SegNet
# ================================
class SegNet(nn.Module):
    """经典SegNet - 使用池化索引的编码器-解码器架构"""

    def __init__(self, n_classes=1):
        super(SegNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, 3, padding=1)
        )

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        # Encoder with pooling indices
        x1 = self.enc1(x)
        x1_size = x1.size()
        x, idx1 = self.pool(x1)

        x2 = self.enc2(x)
        x2_size = x2.size()
        x, idx2 = self.pool(x2)

        x3 = self.enc3(x)
        x3_size = x3.size()
        x, idx3 = self.pool(x3)

        x4 = self.enc4(x)
        x4_size = x4.size()
        x, idx4 = self.pool(x4)

        # Decoder with unpooling
        x = self.unpool(x, idx4, output_size=x4_size)
        x = self.dec4(x)

        x = self.unpool(x, idx3, output_size=x3_size)
        x = self.dec3(x)

        x = self.unpool(x, idx2, output_size=x2_size)
        x = self.dec2(x)

        x = self.unpool(x, idx1, output_size=x1_size)
        x = self.dec1(x)

        return torch.sigmoid(x)


# ================================
# 算法2: PSPNet (Pyramid Scene Parsing Network)
# ================================
class PyramidPooling(nn.Module):
    """金字塔池化模块"""

    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
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


class PSPNet(nn.Module):
    """PSPNet - 金字塔场景解析网络"""

    def __init__(self, n_classes=1):
        super(PSPNet, self).__init__()

        # Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Pyramid Pooling Module
        self.ppm = PyramidPooling(512, pool_sizes=[1, 2, 3, 6])

        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_classes, 1)
        )

    def forward(self, x):
        x_size = x.size()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.ppm(x)
        x = self.final_conv(x)

        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=False)

        return torch.sigmoid(x)


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


class GlobalFeatureExtractor(nn.Module):
    """全局特征提取器"""

    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()

        # Bottleneck blocks
        self.block1 = self._make_bottleneck(64, 64, 3, 1)
        self.block2 = self._make_bottleneck(64, 96, 3, 2)
        self.block3 = self._make_bottleneck(96, 128, 3, 1)

        # Pyramid pooling
        self.ppm = PyramidPooling(128, pool_sizes=[1, 2, 4])

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
        x = self.ppm(x)
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

        # Global feature extractor: 1/32 -> 1/8 with PPM
        x_high = self.global_feature_extractor(x_low)

        # Feature fusion
        x = self.feature_fusion(x_high, x_low)

        # Classifier
        x = self.classifier(x)

        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(x)


# ================================
# 算法4: LinkNet
# ================================
class LinkNetEncoder(nn.Module):
    """LinkNet编码器块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(LinkNetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LinkNetDecoder(nn.Module):
    """LinkNet解码器块"""

    def __init__(self, in_channels, out_channels):
        super(LinkNetDecoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)

        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                         stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.deconv(out)))
        out = F.relu(self.bn3(self.conv2(out)))
        return out


class LinkNet(nn.Module):
    """LinkNet - 轻量级实时分割网络"""

    def __init__(self, n_classes=1):
        super(LinkNet, self).__init__()

        # Initial block
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Encoder
        self.encoder1 = LinkNetEncoder(64, 64)
        self.encoder2 = LinkNetEncoder(64, 128, stride=2)
        self.encoder3 = LinkNetEncoder(128, 256, stride=2)
        self.encoder4 = LinkNetEncoder(256, 512, stride=2)

        # Decoder
        self.decoder4 = LinkNetDecoder(512, 256)
        self.decoder3 = LinkNetDecoder(256, 128)
        self.decoder2 = LinkNetDecoder(128, 64)
        self.decoder1 = LinkNetDecoder(64, 64)

        # Final deconv and classifier
        self.final_deconv = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, n_classes, 2, stride=2),
        )

    def forward(self, x):
        # Initial
        x = self.init_conv(x)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with skip connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final upsampling
        out = self.final_deconv(d1)
        out = self.final_conv(out)

        return torch.sigmoid(out)


# ================================
# 算法5: ENet (Efficient Neural Network)
# ================================
class InitialBlock(nn.Module):
    """ENet初始块"""

    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.maxpool(x)
        out = torch.cat([conv_out, pool_out], dim=1)
        out = self.bn(out)
        return F.relu(out)


class BottleneckBlock(nn.Module):
    """ENet瓶颈块"""

    def __init__(self, in_channels, out_channels, dilation=1, asymmetric=False, downsample=False, dropout_prob=0.1):
        super(BottleneckBlock, self).__init__()

        self.downsample = downsample
        internal_channels = in_channels // 4

        if downsample:
            self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Main branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 1, stride=2 if downsample else 1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True)
        )

        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, (5, 1), padding=(2, 0), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(internal_channels, internal_channels, (1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.ReLU(inplace=True)
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_prob)
        )

    def forward(self, x):
        identity = x

        if self.downsample:
            identity, _ = self.maxpool(identity)
            identity = self.conv_down(identity)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += identity
        return F.relu(out)


class ENet(nn.Module):
    """ENet - 高效实时网络"""

    def __init__(self, n_classes=1):
        super(ENet, self).__init__()

        # Initial
        self.initial = InitialBlock(3, 16)

        # Encoder
        self.encoder1 = nn.Sequential(
            BottleneckBlock(16, 64, downsample=True, dropout_prob=0.01),
            BottleneckBlock(64, 64, dropout_prob=0.01),
            BottleneckBlock(64, 64, dropout_prob=0.01),
            BottleneckBlock(64, 64, dropout_prob=0.01)
        )

        self.encoder2 = nn.Sequential(
            BottleneckBlock(64, 128, downsample=True),
            BottleneckBlock(128, 128),
            BottleneckBlock(128, 128, dilation=2),
            BottleneckBlock(128, 128, asymmetric=True),
            BottleneckBlock(128, 128, dilation=4),
            BottleneckBlock(128, 128),
            BottleneckBlock(128, 128, dilation=8),
            BottleneckBlock(128, 128, asymmetric=True),
            BottleneckBlock(128, 128, dilation=16)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, n_classes, 2, stride=2)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.decoder(x)

        return torch.sigmoid(x)


# ================================
# 评估和训练代码
# ================================
class ModelEvaluator:
    """模型评估器"""

    def __init__(self, device):
        self.device = device

    def calculate_metrics(self, pred, target, threshold=0.5):
        """计算评估指标"""
        pred_binary = (pred > threshold).cpu().numpy().flatten()
        target_binary = target.cpu().numpy().flatten()

        # 基础指标
        accuracy = accuracy_score(target_binary, pred_binary)

        # IoU计算
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        iou = intersection / (union + 1e-8)

        # Precision, Recall, F1
        tp = intersection
        fp = np.sum(pred_binary) - tp
        fn = np.sum(target_binary) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'accuracy': accuracy,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def train_model(self, model, train_loader, val_loader, epochs=20, lr=1e-4):
        """训练模型"""
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_f1': [],
            'val_accuracy': []
        }

        best_iou = 0

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)

                # 确保尺寸匹配
                if outputs.shape != masks.shape:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0
            val_metrics = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = model(images)

                    # 确保尺寸匹配
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                    # 计算验证损失
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    # 计算指标
                    for i in range(outputs.shape[0]):
                        metrics = self.calculate_metrics(outputs[i, 0], masks[i, 0])
                        val_metrics.append(metrics)

            # 计算平均值
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = np.mean([m['iou'] for m in val_metrics])
            avg_val_f1 = np.mean([m['f1_score'] for m in val_metrics])
            avg_val_accuracy = np.mean([m['accuracy'] for m in val_metrics])

            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_iou'].append(avg_val_iou)
            history['val_f1'].append(avg_val_f1)
            history['val_accuracy'].append(avg_val_accuracy)

            scheduler.step(avg_val_loss)

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou

            if epoch % 5 == 0:
                print(f'Epoch {epoch:2d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                      f'IoU: {avg_val_iou:.4f}, F1: {avg_val_f1:.4f}')

        return {'best_iou': best_iou, 'history': history}

    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_metrics = []
        inference_times = []

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                # 测量推理时间
                start_time = time.time()
                outputs = model(images)

                # 确保尺寸匹配
                if outputs.shape != masks.shape:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                inference_time = time.time() - start_time
                inference_times.append(inference_time / images.shape[0])

                # 计算指标
                for i in range(outputs.shape[0]):
                    metrics = self.calculate_metrics(outputs[i, 0], masks[i, 0])
                    all_metrics.append(metrics)

        # 聚合结果
        results = {}
        for key in all_metrics[0].keys():
            results[f'mean_{key}'] = np.mean([m[key] for m in all_metrics])
            results[f'std_{key}'] = np.std([m[key] for m in all_metrics])

        results['avg_inference_time'] = np.mean(inference_times)
        results['total_samples'] = len(all_metrics)

        return results


def prepare_dataset(images_dir, labels_dir, batch_size=4):
    """准备数据集"""
    image_files = []
    label_files = []

    for img_file in sorted(os.listdir(images_dir)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.json")

            if os.path.exists(label_path):
                image_files.append(img_path)
                label_files.append(label_path)

    print(f"Found {len(image_files)} valid image-label pairs")

    if len(image_files) == 0:
        return None

    # 数据划分
    split_idx = int(0.8 * len(image_files))
    train_imgs, val_imgs = image_files[:split_idx], image_files[split_idx:]
    train_labels, val_labels = label_files[:split_idx], label_files[split_idx:]

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = CoastalDataset(train_imgs, train_labels, transform=transform)
    val_dataset = CoastalDataset(val_imgs, val_labels, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def plot_training_curves(training_histories, save_path='./training_curves_rs.png'):
    """绘制训练曲线"""
    if not training_histories:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves - 5 Efficient Remote Sensing Algorithms', fontsize=16, fontweight='bold')

    colors = {
        'SegNet': '#4ECDC4',
        'PSPNet': '#45B7D1',
        'Fast-SCNN': '#96CEB4',
        'LinkNet': '#FFEAA7',
        'ENet': '#DDA15E'
    }

    # 训练损失
    ax1 = axes[0, 0]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], color=colors.get(model_name, 'gray'),
                 label=model_name, linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 验证损失
    ax2 = axes[0, 1]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'], color=colors.get(model_name, 'gray'),
                 label=model_name, linewidth=2, marker='s', markersize=3, alpha=0.8)
    ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 验证IoU
    ax3 = axes[1, 0]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['val_iou']) + 1)
        ax3.plot(epochs, history['val_iou'], color=colors.get(model_name, 'gray'),
                 label=model_name, linewidth=2, marker='^', markersize=3, alpha=0.8)
    ax3.set_title('Validation IoU', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 验证F1
    ax4 = axes[1, 1]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['val_f1']) + 1)
        ax4.plot(epochs, history['val_f1'], color=colors.get(model_name, 'gray'),
                 label=model_name, linewidth=2, marker='d', markersize=3, alpha=0.8)
    ax4.set_title('Validation F1-Score', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1-Score')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Training curves saved to {save_path}")


def plot_comparison(results, save_path='./rs_comparison.png'):
    """绘制对比结果"""
    if not results:
        return

    methods = list(results.keys())
    metrics = ['mean_iou', 'mean_f1_score', 'mean_accuracy']
    metric_names = ['IoU', 'F1-Score', 'Accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('5 Efficient Remote Sensing Algorithms - Performance Comparison',
                 fontsize=14, fontweight='bold')

    colors_map = {
        'SegNet': '#4ECDC4',
        'PSPNet': '#45B7D1',
        'Fast-SCNN': '#96CEB4',
        'LinkNet': '#FFEAA7',
        'ENet': '#DDA15E'
    }

    bar_colors = [colors_map.get(m, 'gray') for m in methods]

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[method][metric] for method in methods]

        bars = axes[i].bar(methods, values, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[i].set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(name, fontsize=11)
        axes[i].tick_params(axis='x', rotation=30, labelsize=10)
        axes[i].set_ylim([0, 1.0])

        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Comparison plot saved to {save_path}")


def main():
    """主对比函数"""
    print("=" * 85)
    print("🌊 Coastal Water Segmentation: 5 Efficient Remote Sensing Algorithms")
    print("=" * 85)
    print("Algorithms: SegNet | PSPNet | Fast-SCNN | LinkNet | ENet")
    print("=" * 85)

    # 数据集路径
    images_dir = "./labelme_images/converted"
    labels_dir = "./labelme_images/annotations/"

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("❌ Dataset directories not found. Please check paths.")
        return

    # 准备数据集
    data_loaders = prepare_dataset(images_dir, labels_dir, batch_size=2)
    if data_loaders is None:
        return

    train_loader, val_loader = data_loaders

    # 初始化5个模型
    models = {
        'SegNet': SegNet(n_classes=1).to(device),
        'PSPNet': PSPNet(n_classes=1).to(device),
        'Fast-SCNN': FastSCNN(n_classes=1).to(device),
        'LinkNet': LinkNet(n_classes=1).to(device),
        'ENet': ENet(n_classes=1).to(device)
    }

    # 打印模型参数数量
    print("\n📊 Model Parameters:")
    print("-" * 50)
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  {name:<12}: {param_count:>12,} parameters ({param_count / 1e6:.2f}M)")
    print("-" * 50)

    evaluator = ModelEvaluator(device)
    results = {}
    training_histories = {}

    # 训练和评估每个模型
    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"🚀 Training {name}...")
        print(f"{'=' * 60}")

        # 根据模型调整训练轮数
        if name == 'SegNet':
            epochs = 15
        elif name == 'Fast-SCNN':
            epochs = 25
        else:
            epochs = 20

        training_results = evaluator.train_model(model, train_loader, val_loader, epochs=epochs, lr=1e-4)
        training_histories[name] = training_results['history']

        print(f"\n✅ Best IoU during training: {training_results['best_iou']:.4f}")

        # 评估模型
        print(f"📈 Final evaluation on validation set...")
        eval_results = evaluator.evaluate_model(model, val_loader)
        results[name] = eval_results

        print(f"\n🎯 {name} Final Results:")
        print(f"  • IoU:      {eval_results['mean_iou']:.4f} ± {eval_results['std_iou']:.4f}")
        print(f"  • F1-Score: {eval_results['mean_f1_score']:.4f} ± {eval_results['std_f1_score']:.4f}")
        print(f"  • Accuracy: {eval_results['mean_accuracy']:.4f} ± {eval_results['std_accuracy']:.4f}")
        print(f"  • Inference Time: {eval_results['avg_inference_time'] * 1000:.2f}ms per image")

    # 绘制训练曲线
    print(f"\n{'=' * 60}")
    print("📉 Generating training curves...")
    plot_training_curves(training_histories)

    # 显示对比结果
    print(f"\n{'=' * 80}")
    print("🏆 FINAL COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Algorithm':<12} {'IoU':<10} {'F1-Score':<11} {'Accuracy':<11} {'Params':<12} {'Time(ms)':<10}")
    print("-" * 80)

    for method_name, result in results.items():
        param_count = sum(p.numel() for p in models[method_name].parameters())

        print(f"{method_name:<12} "
              f"{result['mean_iou']:<10.4f} "
              f"{result['mean_f1_score']:<11.4f} "
              f"{result['mean_accuracy']:<11.4f} "
              f"{param_count / 1e6:<12.2f}M "
              f"{result['avg_inference_time'] * 1000:<10.2f}")

    # 获胜者分析
    print(f"\n{'=' * 80}")
    print("🎖️  WINNER ANALYSIS:")
    print("-" * 80)


    best_iou = max(results.items(), key=lambda x: x[1]['mean_iou'])
    best_f1 = max(results.items(), key=lambda x: x[1]['mean_f1_score'])
    best_acc = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
    fastest = min(results.items(), key=lambda x: x[1]['avg_inference_time'])
    lightest = min(models.items(), key=lambda x: sum(p.numel() for p in x[1].parameters()))

    print(f"  🥇 Best IoU:        {best_iou[0]:<12} ({best_iou[1]['mean_iou']:.4f})")
    print(f"  🥇 Best F1-Score:   {best_f1[0]:<12} ({best_f1[1]['mean_f1_score']:.4f})")
    print(f"  🥇 Best Accuracy:   {best_acc[0]:<12} ({best_acc[1]['mean_accuracy']:.4f})")
    print(f"  ⚡ Fastest:         {fastest[0]:<12} ({fastest[1]['avg_inference_time'] * 1000:.2f}ms)")
    print(
        f"  🪶 Lightest:        {lightest[0]:<12} ({sum(p.numel() for p in lightest[1].parameters()) / 1e6:.2f}M params)")

    # 绘制对比图
    print(f"\n📊 Generating comparison plots...")
    plot_comparison(results)

    print(f"\n{'=' * 80}")
    print("✅ Comparison complete!")
    print(f"{'=' * 80}")
    print(f"📁 Results saved to:")
    print(f"   • ./training_curves_rs.png (Training history)")
    print(f"   • ./rs_comparison.png (Performance comparison)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()