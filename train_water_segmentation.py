#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海水区域语义分割模型训练程序
基于labelme标注的海水区域训练语义分割模型

作者: CoastSat海岸线提取助手
创建日期: 2025-01-26
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datetime import datetime
import pickle
from osgeo import gdal
import warnings
warnings.filterwarnings('ignore')

class WaterSegmentationDataset(Dataset):
    """海水区域分割数据集"""
    
    def __init__(self, image_paths, label_paths, transform=None, image_size=(512, 512)):
        """
        初始化数据集
        
        参数:
        image_paths: list, 图像文件路径列表
        label_paths: list, 标注文件路径列表
        transform: torchvision.transforms, 数据变换
        image_size: tuple, 输入图像尺寸
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像（支持多种格式）
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # 读取标注并生成mask
        mask = self.create_mask_from_labelme(self.label_paths[idx], image.size)
        
        # 调整尺寸
        image = image.resize(self.image_size)
        mask = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)
        mask = np.array(mask)
        
        # 数据变换
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def load_image(self, image_path):
        """
        加载图像 - 针对converted文件夹中的PNG和原始TIF进行优化
        
        参数:
        image_path: str, 图像文件路径
        
        返回:
        PIL.Image: RGB图像
        """
        try:
            if image_path.lower().endswith(('.tif', '.tiff')):
                # 处理原始TIF格式（与tif_to_image.py保持一致的水体增强）
                return self.load_tif_with_water_enhancement(image_path)
            else:
                # 常规图像格式，包括从TIF转换后的PNG
                # 这些PNG已经经过了水体增强处理
                return Image.open(image_path).convert('RGB')
                
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            # 返回空白图像
            return Image.new('RGB', (512, 512), (0, 0, 0))
    
    def load_tif_with_water_enhancement(self, tif_path):
        """
        加载TIF文件并应用水体增强（与tif_to_image.py保持一致）
        
        参数:
        tif_path: str, TIF文件路径
        
        返回:
        PIL.Image: RGB图像
        """
        dataset = gdal.Open(tif_path)
        if dataset is None:
            raise ValueError(f"无法打开TIF文件: {tif_path}")
        
        # 读取波段数据（与转换程序保持一致）
        bands = []
        for i in range(1, min(dataset.RasterCount + 1, 7)):  # 最多读取6个波段
            band = dataset.GetRasterBand(i)
            data = band.ReadAsArray()
            bands.append(data)
        
        bands = np.array(bands)
        
        # 创建RGB图像（与tif_to_image.py的逻辑一致）
        if bands.shape[0] >= 3:
            if bands.shape[0] >= 4:
                # 使用NIR-Red-Green组合突出水体（与转换程序一致）
                try:
                    rgb = np.dstack([bands[4], bands[3], bands[2]])  # NIR, Red, Green
                except IndexError:
                    rgb = np.dstack([bands[2], bands[1], bands[0]])  # 标准RGB
            else:
                rgb = np.dstack([bands[2], bands[1], bands[0]])  # Red, Green, Blue
        else:
            # 灰度图像
            gray = bands[0]
            rgb = np.dstack([gray, gray, gray])
        
        # 图像增强（与转换程序一致）
        rgb_enhanced = self.enhance_image_for_water(rgb)
        return Image.fromarray(rgb_enhanced.astype(np.uint8))
    
    def enhance_image_for_water(self, rgb):
        """
        增强图像对比度，突出水体区域（与tif_to_image.py保持一致）
        
        参数:
        rgb: numpy.ndarray, RGB图像数组
        
        返回:
        numpy.ndarray: 增强后的图像
        """
        enhanced = np.zeros_like(rgb)
        
        for i in range(rgb.shape[2]):
            band = rgb[:, :, i]
            
            # 计算百分位数进行拉伸
            p2, p98 = np.percentile(band, [2, 98])
            
            # 线性拉伸
            band_stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            
            # 对水体区域进行额外增强
            if i == 0:  # 假设第一个波段是近红外或红色
                # 增强低值区域（可能是水体）
                mask = band_stretched < 100
                band_stretched[mask] = band_stretched[mask] * 0.7  # 降低亮度突出水体
            
            enhanced[:, :, i] = band_stretched
        
        return enhanced
    
    def create_mask_from_labelme(self, label_path, image_size):
        """
        从labelme标注文件创建分割mask
        
        参数:
        label_path: str, labelme标注文件路径
        image_size: tuple, 图像尺寸 (width, height)
        
        返回:
        numpy.ndarray: 分割mask (0-其他, 1-海水)
        """
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 创建空白mask
            mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
            
            # 遍历所有形状
            for shape in label_data.get('shapes', []):
                if shape['label'].lower() in ['water', 'sea', '海水', '水体']:
                    # 获取多边形点
                    points = np.array(shape['points'], dtype=np.int32)
                    
                    # 填充多边形区域
                    cv2.fillPoly(mask, [points], 1)
            
            return mask
            
        except Exception as e:
            print(f"读取标注文件失败 {label_path}: {e}")
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

class UNet(nn.Module):
    """U-Net语义分割网络"""
    
    def __init__(self, n_channels=3, n_classes=2):
        """
        初始化U-Net
        
        参数:
        n_channels: int, 输入通道数
        n_classes: int, 分类类别数
        """
        super(UNet, self).__init__()
        
        # 编码器
        self.enc1 = self.conv_block(n_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)
        
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # 输出层
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def conv_block(self, in_channels, out_channels):
        """卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # 解码路径
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

class WaterSegmentationTrainer:
    """海水分割模型训练器 - 集成CoastSat训练思想"""
    
    def __init__(self, device='cpu'):
        """
        初始化训练器
        
        参数:
        device: str, 计算设备
        """
        self.device = device
        self.model = UNet(n_channels=3, n_classes=2).to(device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # 学习率调度器（CoastSat风格）
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 训练历史记录（参考CoastSat的记录方式）
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'accuracies': [],
            'iou_scores': [],
            'best_model_epoch': 0,
            'training_time': 0
        }
    
    def calculate_iou(self, pred_mask, true_mask):
        """
        计算IoU分数（参考CoastSat的评估方法）
        
        参数:
        pred_mask: torch.Tensor, 预测mask
        true_mask: torch.Tensor, 真实mask
        
        返回:
        float: IoU分数
        """
        intersection = torch.logical_and(pred_mask, true_mask).sum().float()
        union = torch.logical_or(pred_mask, true_mask).sum().float()
        
        if union == 0:
            return 1.0
        
        return (intersection / union).item()
    
    def validate_model(self, val_loader):
        """
        验证模型性能（集成CoastSat的验证思想）
        
        参数:
        val_loader: DataLoader, 验证数据加载器
        
        返回:
        dict: 验证结果
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # 计算预测结果
                pred_masks = torch.argmax(outputs, dim=1)
                
                # 计算指标
                accuracy = (pred_masks == masks).float().mean()
                iou = self.calculate_iou(pred_masks == 1, masks == 1)
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_iou += iou
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'iou': total_iou / num_batches
        }
    
    def create_training_visualization(self, epoch, train_loss, val_metrics, save_dir):
        """
        创建训练过程可视化（参考CoastSat的可视化风格）
        
        参数:
        epoch: int, 当前轮次
        train_loss: float, 训练损失
        val_metrics: dict, 验证指标
        save_dir: str, 保存目录
        """
        # 确保中文字体正确显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.training_history['train_losses'], 'b-', label='训练损失')
        axes[0, 0].plot(self.training_history['val_losses'], 'r-', label='验证损失')
        axes[0, 0].set_title('训练损失变化')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(self.training_history['accuracies'], 'g-', label='验证准确率')
        axes[0, 1].set_title('模型准确率变化')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU分数曲线
        axes[1, 0].plot(self.training_history['iou_scores'], 'm-', label='IoU分数')
        axes[1, 0].set_title('IoU分数变化')
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('IoU分数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[1, 1].plot(self.training_history['learning_rates'], 'orange', label='学习率')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('学习率')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.suptitle(f'训练进度 - 第 {epoch+1} 轮', fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        viz_path = os.path.join(save_dir, f'training_progress_epoch_{epoch+1}.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrix(self, val_loader, save_dir, epoch):
        """
        创建混淆矩阵（参考CoastSat的评估方法）
        
        参数:
        val_loader: DataLoader, 验证数据加载器
        save_dir: str, 保存目录
        epoch: int, 当前轮次
        """
        # 确保中文字体正确显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        self.model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                pred_masks = torch.argmax(outputs, dim=1)
                
                all_preds.extend(pred_masks.cpu().numpy().flatten())
                all_true.extend(masks.cpu().numpy().flatten())
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_true, all_preds)
        
        # 绘制混淆矩阵
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = ['其他', '海水']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=f'混淆矩阵 - 第 {epoch+1} 轮',
               ylabel='真实标签',
               xlabel='预测标签')
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        cm_path = os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self, train_loader, val_loader, epochs=200, save_dir="./models"):
        """
        训练模型（集成CoastSat的训练策略）
        
        参数:
        train_loader: DataLoader, 训练数据加载器
        val_loader: DataLoader, 验证数据加载器
        epochs: int, 训练轮数
        save_dir: str, 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 记录训练开始时间
        start_time = datetime.now()
        
        best_val_loss = float('inf')
        best_iou = 0.0
        patience_counter = 0
        max_patience = 20  # 早停耐心值
        
        print(f"\n=== 开始训练海水分割模型 ===")
        print(f"训练设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {train_loader.batch_size}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print(f"保存目录: {save_dir}\n")
        
        for epoch in range(epochs):
            epoch_start_time = datetime.now()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            print(f"轮次 {epoch+1}/{epochs}")
            print("-" * 50)
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # 每10个批次打印一次进度
                if batch_idx % 10 == 0:
                    print(f'  批次 {batch_idx+1}/{len(train_loader)}, 损失: {loss.item():.4f}')
            
            train_loss /= train_batches
            
            # 验证阶段
            val_metrics = self.validate_model(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_metrics['loss'])
            self.training_history['accuracies'].append(val_metrics['accuracy'])
            self.training_history['iou_scores'].append(val_metrics['iou'])
            self.training_history['learning_rates'].append(current_lr)
            
            # 计算轮次耗时
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            
            # 打印轮次结果
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_metrics['loss']:.4f}")
            print(f"  验证准确率: {val_metrics['accuracy']:.4f}")
            print(f"  IoU分数: {val_metrics['iou']:.4f}")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  轮次耗时: {epoch_time:.1f}秒")
            
            # 保存最佳模型（基于IoU分数）
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                best_val_loss = val_metrics['loss']
                self.training_history['best_model_epoch'] = epoch
                patience_counter = 0
                
                # 保存最佳模型
                model_path = os.path.join(save_dir, 'best_water_segmentation_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"  ✓ 保存最佳模型 (IoU: {best_iou:.4f})")
            else:
                patience_counter += 1
                print(f"  等待改进... ({patience_counter}/{max_patience})")
            
            # 创建训练可视化
            if (epoch + 1) % 5 == 0 or epoch == 0:  # 每5轮或第一轮创建可视化
                self.create_training_visualization(epoch, train_loss, val_metrics, save_dir)
                self.create_confusion_matrix(val_loader, save_dir, epoch)
            
            # 早停检查
            if patience_counter >= max_patience:
                print(f"\n早停触发！已连续 {max_patience} 轮无改进")
                break
            
            print(f"  当前最佳IoU: {best_iou:.4f} (第 {self.training_history['best_model_epoch']+1} 轮)\n")
        
        # 训练完成
        total_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = total_time
        
        print("=" * 60)
        print("🎉 训练完成！")
        print(f"总耗时: {total_time//60:.0f}分 {total_time%60:.0f}秒")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"最佳IoU分数: {best_iou:.4f}")
        print(f"最佳模型轮次: {self.training_history['best_model_epoch']+1}")
        print(f"模型保存位置: {os.path.join(save_dir, 'best_water_segmentation_model.pth')}")
        
        # 保存完整训练历史
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        # 创建最终训练报告
        self.create_final_training_report(save_dir)
        
        print("=" * 60)
        
        return self.training_history
    
    def create_final_training_report(self, save_dir):
        """
        创建最终训练报告（参考CoastSat的报告风格）
        
        参数:
        save_dir: str, 保存目录
        """
        # 确保中文字体正确显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 损失对比图
        axes[0, 0].plot(self.training_history['train_losses'], 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(self.training_history['val_losses'], 'r-', label='验证损失', linewidth=2)
        axes[0, 0].axvline(x=self.training_history['best_model_epoch'], color='g', 
                          linestyle='--', alpha=0.7, label='最佳模型')
        axes[0, 0].set_title('损失变化曲线', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(self.training_history['accuracies'], 'g-', linewidth=2)
        axes[0, 1].axvline(x=self.training_history['best_model_epoch'], color='r', 
                          linestyle='--', alpha=0.7, label='最佳模型')
        axes[0, 1].set_title('验证准确率', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU分数曲线
        axes[0, 2].plot(self.training_history['iou_scores'], 'm-', linewidth=2)
        axes[0, 2].axvline(x=self.training_history['best_model_epoch'], color='r', 
                          linestyle='--', alpha=0.7, label='最佳模型')
        axes[0, 2].set_title('IoU分数变化', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('训练轮次')
        axes[0, 2].set_ylabel('IoU分数')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[1, 0].plot(self.training_history['learning_rates'], 'orange', linewidth=2)
        axes[1, 0].set_title('学习率调整', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_ylabel('学习率')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 训练统计信息
        axes[1, 1].axis('off')
        stats_text = f"""
        训练统计信息
        
        总训练轮次: {len(self.training_history['train_losses'])}
        最佳模型轮次: {self.training_history['best_model_epoch'] + 1}
        最佳验证损失: {min(self.training_history['val_losses']):.4f}
        最佳IoU分数: {max(self.training_history['iou_scores']):.4f}
        最佳准确率: {max(self.training_history['accuracies']):.4f}
        训练总耗时: {self.training_history['training_time']//60:.0f}分{self.training_history['training_time']%60:.0f}秒
        """
        axes[1, 1].text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 性能分布
        final_metrics = [
            self.training_history['val_losses'][-1],
            self.training_history['accuracies'][-1],
            self.training_history['iou_scores'][-1]
        ]
        metric_names = ['验证损失', '准确率', 'IoU分数']
        colors = ['red', 'green', 'magenta']
        
        axes[1, 2].bar(metric_names, final_metrics, color=colors, alpha=0.7)
        axes[1, 2].set_title('最终模型性能', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('分数')
        
        # 添加数值标签
        for i, v in enumerate(final_metrics):
            axes[1, 2].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('🌊 海水分割模型训练报告', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存报告
        report_path = os.path.join(save_dir, 'final_training_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练报告已保存: {report_path}")
    
    def prepare_dataset(self, images_dir, labels_dir):
        """
        准备训练数据集
        
        参数:
        images_dir: str, 图像目录
        labels_dir: str, 标注目录
        
        返回:
        tuple: (train_loader, val_loader)
        """
        # 获取图像和标注文件对
        image_files = []
        label_files = []
        
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                img_path = os.path.join(images_dir, img_file)
                
                # 查找对应的标注文件
                base_name = os.path.splitext(img_file)[0]
                label_path = os.path.join(labels_dir, f"{base_name}.json")
                
                if os.path.exists(label_path):
                    image_files.append(img_path)
                    label_files.append(label_path)
        
        print(f"找到 {len(image_files)} 对图像-标注文件")
        
        if len(image_files) == 0:
            raise ValueError("未找到匹配的图像-标注文件对")
        
        # 数据质量检查
        print("\n=== 数据质量检查 ===")
        valid_pairs = []
        for img_path, label_path in zip(image_files, label_files):
            try:
                # 检查图像
                img = Image.open(img_path)
                if img.size[0] < 50 or img.size[1] < 50:
                    print(f"⚠️  跳过尺寸过小的图像: {os.path.basename(img_path)}")
                    continue
                
                # 检查标注
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                
                # 检查是否有海水标注
                has_water = False
                for shape in label_data.get('shapes', []):
                    if shape['label'].lower() in ['water', 'sea', '海水', '水体']:
                        has_water = True
                        break
                
                if has_water:
                    valid_pairs.append((img_path, label_path))
                else:
                    print(f"⚠️  跳过无海水标注的文件: {os.path.basename(label_path)}")
                    
            except Exception as e:
                print(f"⚠️  跳过损坏的文件对: {os.path.basename(img_path)} - {e}")
        
        if len(valid_pairs) == 0:
            raise ValueError("没有找到有效的图像-标注文件对")
        
        image_files, label_files = zip(*valid_pairs)
        print(f"✓ 有效文件对: {len(image_files)}")
        
        # 划分训练和验证集
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            image_files, label_files, test_size=0.2, random_state=42, shuffle=True
        )
        
        # 创建数据集
        train_dataset = WaterSegmentationDataset(
            train_imgs, train_labels, transform=self.train_transform
        )
        val_dataset = WaterSegmentationDataset(
            val_imgs, val_labels, transform=self.val_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        print(f"✓ 训练集: {len(train_dataset)} 样本")
        print(f"✓ 验证集: {len(val_dataset)} 样本")
        print("=" * 40)
        
        return train_loader, val_loader

def main():
    """主训练函数"""
    print("=== 海水区域语义分割模型训练程序 ===")
    print("基于labelme标注的海水区域训练语义分割模型")
    print("\n📋 推荐工作流程:")
    print("1. 使用 tif_to_image.py 将TIF文件转换为PNG (./labelme_images/converted/)")
    print("2. 使用 labelme 对PNG图像进行标注 (保存到 ./labelme_images/annotations/)")
    print("3. 运行本程序进行模型训练")
    print("4. 使用 predict_coastline.py 进行海岸线预测")
    
    # 训练模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    trainer = WaterSegmentationTrainer(device=device)
    
    images_dir = input("请输入图像目录路径 (默认: ./labelme_images/converted): ").strip()
    if not images_dir:
        images_dir = "./labelme_images/converted"
    
    labels_dir = input("请输入标注目录路径 (默认: ./labelme_images/annotations/): ").strip()
    if not labels_dir:
        labels_dir = "./labelme_images/annotations/"
    
    try:
        train_loader, val_loader = trainer.prepare_dataset(images_dir, labels_dir)
        print("\n=== 开始训练模型 ===")
        print("训练参数:")
        print(f"- 图像目录: {images_dir}")
        print(f"- 标注目录: {labels_dir}")
        print(f"- 训练集: {len(train_loader.dataset)} 样本")
        print(f"- 验证集: {len(val_loader.dataset)} 样本")
        print(f"- 计算设备: {device}")
        
        epochs = int(input("请输入训练轮数 (默认: 200): ") or "200")
        save_dir = input("请输入模型保存目录 (默认: ./models): ").strip() or "./models"
        
        trainer.train(train_loader, val_loader, epochs=epochs, save_dir=save_dir)
        
    except Exception as e:
        print(f"训练失败: {e}")

if __name__ == "__main__":
    main()
