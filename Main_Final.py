#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海岸线检测算法对比: DeepLabV3+ vs YOLO-SEG vs Robust UNet
包含重新实现的Robust UNet
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


# Robust UNet Components
class ChannelAttention(nn.Module):
    """Channel Attention Module"""

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_att)
        return x * self.sigmoid(x_att)


class AttentionGate(nn.Module):
    """Attention Gate for Skip Connections"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualBlock(nn.Module):
    """Residual Block with improved normalization"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

        # Channel attention
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply attention
        out = self.ca(out)
        out = self.sa(out)

        out += residual
        out = self.relu(out)

        return out


class DilatedBlock(nn.Module):
    """Dilated Convolution Block for capturing multi-scale features"""

    def __init__(self, in_channels, out_channels):
        super(DilatedBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=4, dilation=4)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.bn(out)
        out = self.relu(out)

        return out


class RobustUNet(nn.Module):
    """Robust UNet with attention mechanisms and improved architecture"""

    def __init__(self, n_channels=3, n_classes=1, base_channels=64):
        super(RobustUNet, self).__init__()

        # Encoder
        self.inc = ResidualBlock(n_channels, base_channels, dropout_rate=0.1)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(base_channels, base_channels * 2, dropout_rate=0.1)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(base_channels * 2, base_channels * 4, dropout_rate=0.2)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(base_channels * 4, base_channels * 8, dropout_rate=0.2)
        )

        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DilatedBlock(base_channels * 8, base_channels * 16),
            ResidualBlock(base_channels * 16, base_channels * 16, dropout_rate=0.3)
        )

        # Attention gates
        self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels // 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 8, dropout_rate=0.2)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4, dropout_rate=0.2)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2, dropout_rate=0.1)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, dropout_rate=0.1)

        # Final output
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, n_classes, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)  # 64, 512, 512
        x2 = self.down1(x1)  # 128, 256, 256
        x3 = self.down2(x2)  # 256, 128, 128
        x4 = self.down3(x3)  # 512, 64, 64

        # Bottleneck
        x5 = self.bottleneck(x4)  # 1024, 32, 32

        # Decoder path with attention
        x = self.up4(x5)
        x4_att = self.att4(x, x4)
        x = torch.cat([x4_att, x], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x3_att = self.att3(x, x3)
        x = torch.cat([x3_att, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x2_att = self.att2(x, x2)
        x = torch.cat([x2_att, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x1_att = self.att1(x, x1)
        x = torch.cat([x1_att, x], dim=1)
        x = self.dec1(x)

        return self.outc(x)


# Keep the same ASPP and DeepLabV3+ classes as before
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for DeepLabV3+"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1)

        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.shape[-2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x5 = self.global_pool(x)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_out(x)
        return F.relu(self.bn(x))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with fixed upsampling"""

    def __init__(self, n_classes=1):
        super(DeepLabV3Plus, self).__init__()

        # Simplified backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 512 -> 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),  # 256 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # ASPP at 32x32
        self.aspp = ASPP(512, 256)

        # Decoder - need to go from 32x32 back to 512x512
        self.decoder = nn.Sequential(
            # 32 -> 64
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 128 -> 256
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 256 -> 512
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final conv to get class predictions
            nn.Conv2d(16, n_classes, 3, padding=1)
        )

    def forward(self, x):
        # Forward through backbone
        x = self.conv1(x)  # 512 -> 256
        x = self.conv2(x)  # 256 -> 128
        x = self.conv3(x)  # 128 -> 64
        x = self.conv4(x)  # 64 -> 32

        # ASPP
        x = self.aspp(x)  # Still 32x32

        # Decoder
        x = self.decoder(x)  # 32 -> 512

        return torch.sigmoid(x)


class YOLOSeg(nn.Module):
    """YOLO-style segmentation network with fixed upsampling"""

    def __init__(self, n_classes=1):
        super(YOLOSeg, self).__init__()

        # YOLO-style backbone
        self.backbone = nn.Sequential(
            # Layer 1: 512 -> 256
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # Layer 2: 256 -> 128
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # Layer 3: 128 -> 64
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # Layer 4: 64 -> 32
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        # Segmentation head - upsample from 32x32 to 512x512
        self.seg_head = nn.Sequential(
            # 32 -> 64
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 -> 256
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            # 256 -> 512
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            # Final prediction
            nn.Conv2d(16, n_classes, 3, padding=1),
        )

    def forward(self, x):
        features = self.backbone(x)  # Should be 32x32
        segmentation = self.seg_head(features)  # Should be 512x512
        return torch.sigmoid(segmentation)


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

    def train_model(self, model, train_loader, val_loader, epochs=25, lr=1e-4):
        """训练模型"""
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

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

            scheduler.step(avg_train_loss)

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou

            if epoch % 5 == 0:
                print(
                    f'Epoch {epoch:2d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, F1: {avg_val_f1:.4f}')

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


def plot_training_curves(training_histories, save_path='./training_curves.png'):
    """绘制训练曲线"""
    if not training_histories:
        return

    # 设置图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')

    # 颜色和线型设置
    colors = {'DeepLabV3+': 'red', 'YOLO-SEG': 'blue', 'Robust UNet': 'green'}
    linestyles = {'DeepLabV3+': '-', 'YOLO-SEG': '--', 'Robust UNet': '-.'}

    # 绘制训练损失
    ax1 = axes[0, 0]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'],
                 color=colors.get(model_name, 'gray'),
                 linestyle=linestyles.get(model_name, '-'),
                 label=model_name, linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制验证损失
    ax2 = axes[0, 1]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'],
                 color=colors.get(model_name, 'gray'),
                 linestyle=linestyles.get(model_name, '-'),
                 label=model_name, linewidth=2, marker='s', markersize=4)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 绘制验证IoU
    ax3 = axes[1, 0]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['val_iou']) + 1)
        ax3.plot(epochs, history['val_iou'],
                 color=colors.get(model_name, 'gray'),
                 linestyle=linestyles.get(model_name, '-'),
                 label=model_name, linewidth=2, marker='^', markersize=4)
    ax3.set_title('Validation IoU')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 绘制验证F1-Score
    ax4 = axes[1, 1]
    for model_name, history in training_histories.items():
        epochs = range(1, len(history['val_f1']) + 1)
        ax4.plot(epochs, history['val_f1'],
                 color=colors.get(model_name, 'gray'),
                 linestyle=linestyles.get(model_name, '-'),
                 label=model_name, linewidth=2, marker='d', markersize=4)
    ax4.set_title('Validation F1-Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1-Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Training curves saved to {save_path}")


def plot_comparison(results):
    """绘制对比结果"""
    if not results:
        return

    methods = list(results.keys())
    metrics = ['mean_iou', 'mean_f1_score', 'mean_accuracy']
    metric_names = ['IoU', 'F1-Score', 'Accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[method][metric] for method in methods]

        colors = ['lightcoral', 'lightblue', 'lightgreen']
        bars = axes[i].bar(methods, values, color=colors)
        axes[i].set_title(f'{name} Comparison')
        axes[i].set_ylabel(name)
        axes[i].tick_params(axis='x', rotation=45)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                         f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('./coastal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主对比函数"""
    print("Coastal Water Segmentation: DeepLabV3+ vs YOLO-SEG vs Robust UNet Comparison")
    print("=" * 75)

    # 数据集路径
    images_dir = "./labelme_images/converted"
    labels_dir = "./labelme_images/annotations/"

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Dataset directories not found. Please check paths.")
        return

    # 准备数据集
    data_loaders = prepare_dataset(images_dir, labels_dir, batch_size=2)
    if data_loaders is None:
        return

    train_loader, val_loader = data_loaders

    # 初始化模型
    models = {
        'Robust UNet': RobustUNet(n_channels=3, n_classes=1).to(device),
        'DeepLabV3+': DeepLabV3Plus(n_classes=1).to(device),
        'YOLO-SEG': YOLOSeg(n_classes=1).to(device)
    }

    # 打印模型参数数量
    print("\nModel Parameters:")
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {param_count:,} parameters")

    evaluator = ModelEvaluator(device)
    results = {}
    training_histories = {}

    # 训练和评估每个模型
    for name, model in models.items():
        print(f"\n{'=' * 40}")
        print(f"Training {name}...")

        if name == 'DeepLabV3+':
            training_results = evaluator.train_model(model, train_loader, val_loader, epochs=25, lr=1e-4)
        else:
            training_results = evaluator.train_model(model, train_loader, val_loader, epochs=20, lr=1e-4)

        training_histories[name] = training_results['history']
        print(f"Best IoU during training: {training_results['best_iou']:.4f}")

        # 评估模型
        eval_results = evaluator.evaluate_model(model, val_loader)
        results[name] = eval_results

        print(f"Final evaluation:")
        print(f"  IoU: {eval_results['mean_iou']:.4f} ± {eval_results['std_iou']:.3f}")
        print(f"  F1-Score: {eval_results['mean_f1_score']:.4f} ± {eval_results['std_f1_score']:.3f}")
        print(f"  Accuracy: {eval_results['mean_accuracy']:.4f} ± {eval_results['std_accuracy']:.3f}")
        print(f"  Inference Time: {eval_results['avg_inference_time'] * 1000:.2f}ms")

    # 绘制训练曲线
    print(f"\n{'=' * 40}")
    print("Generating training curves...")
    plot_training_curves(training_histories)

    # 显示对比结果
    print(f"\n{'=' * 75}")
    print("FINAL COMPARISON RESULTS")
    print(f"{'=' * 75}")
    print(f"{'Method':<15} {'IoU':<10} {'F1-Score':<10} {'Accuracy':<10} {'Parameters':<12} {'Time(ms)':<10}")
    print("-" * 75)

    for method_name, result in results.items():
        param_count = sum(p.numel() for p in models[method_name].parameters())

        print(f"{method_name:<15} "
              f"{result['mean_iou']:.4f}    "
              f"{result['mean_f1_score']:.4f}     "
              f"{result['mean_accuracy']:.4f}     "
              f"{param_count / 1000000:.1f}M        "
              f"{result['avg_inference_time'] * 1000:.2f}")

    print(f"\n🏆 WINNER ANALYSIS:")
    best_iou_method = max(results.items(), key=lambda x: x[1]['mean_iou'])
    best_f1_method = max(results.items(), key=lambda x: x[1]['mean_f1_score'])
    best_acc_method = max(results.items(), key=lambda x: x[1]['mean_accuracy'])

    print(f"  Best IoU: {best_iou_method[0]} ({best_iou_method[1]['mean_iou']:.4f})")
    print(f"  Best F1-Score: {best_f1_method[0]} ({best_f1_method[1]['mean_f1_score']:.4f})")
    print(f"  Best Accuracy: {best_acc_method[0]} ({best_acc_method[1]['mean_accuracy']:.4f})")

    # 绘制对比图
    plot_comparison(results)

    print(f"\nComparison complete!")
    print(f"Results saved to:")
    print(f"  - ./training_curves.png (Training curves)")
    print(f"  - ./coastal_comparison.png (Final comparison)")


if __name__ == "__main__":
    main()