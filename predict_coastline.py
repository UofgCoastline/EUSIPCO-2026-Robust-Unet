#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海水区域语义分割预测与海岸线提取程序
使用训练好的模型进行海水区域分割和海岸线提取

作者: CoastSat海岸线提取助手
创建日期: 2025-01-26
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import Canvas, PhotoImage
from PIL import ImageTk
import threading
from osgeo import gdal
import warnings
import glob
warnings.filterwarnings('ignore')

class ZoomableImageCanvas(tk.Frame):
    """可缩放和拖拽的图像Canvas"""
    
    def __init__(self, parent, image):
        """
        初始化可缩放图像Canvas
        
        参数:
        parent: 父窗口
        image: PIL.Image, 要显示的图像
        """
        super().__init__(parent)
        
        # 图像相关变量
        self.original_image = image
        self.current_image = image
        self.photo = None
        
        # 缩放和拖拽相关变量
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False
        
        # 创建Canvas和滚动条
        self.canvas = tk.Canvas(self, bg='white', highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        # 配置Canvas滚动
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, 
                            xscrollcommand=self.h_scrollbar.set)
        
        # 布局
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定事件
        self.bind_events()
        
        # 初始显示图像
        self.update_image()
    
    def bind_events(self):
        """绑定鼠标和键盘事件"""
        # 鼠标滚轮缩放
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux
        
        # 鼠标拖拽
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        
        # 键盘快捷键
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.focus_set()
        
        # 双击重置缩放
        self.canvas.bind("<Double-Button-1>", self.reset_zoom)
    
    def on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        # 获取鼠标位置
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # 计算缩放因子
        if event.delta > 0 or event.num == 4:
            # 向上滚动，放大
            scale = 1.1
        else:
            # 向下滚动，缩小
            scale = 0.9
        
        # 应用缩放
        self.zoom_at_point(x, y, scale)
    
    def zoom_at_point(self, x, y, scale):
        """在指定点进行缩放"""
        new_scale = self.scale_factor * scale
        
        # 限制缩放范围
        if new_scale < self.min_scale:
            new_scale = self.min_scale
        elif new_scale > self.max_scale:
            new_scale = self.max_scale
        
        if new_scale == self.scale_factor:
            return
        
        # 计算缩放前后的位置差
        scale_change = new_scale / self.scale_factor
        
        # 更新缩放因子
        self.scale_factor = new_scale
        
        # 更新图像
        self.update_image()
        
        # 调整视图位置以保持缩放点居中
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if self.current_image.width > canvas_width:
            center_x = canvas_width / 2
            scroll_x = max(0, min(1, (x * scale_change - center_x) / (self.current_image.width - canvas_width)))
            self.canvas.xview_moveto(scroll_x)
        
        if self.current_image.height > canvas_height:
            center_y = canvas_height / 2
            scroll_y = max(0, min(1, (y * scale_change - center_y) / (self.current_image.height - canvas_height)))
            self.canvas.yview_moveto(scroll_y)
    
    def on_drag_start(self, event):
        """开始拖拽"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_dragging = True
        self.canvas.configure(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)
    
    def on_drag_motion(self, event):
        """拖拽移动"""
        if not self.is_dragging:
            return
        
        # 使用scan_dragto实现平滑拖拽
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        
        # 更新拖拽起始点
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    
    def on_drag_end(self, event):
        """结束拖拽"""
        self.is_dragging = False
        self.canvas.configure(cursor="")
    
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.keysym == 'plus' or event.keysym == 'equal':
            # 放大
            self.zoom_at_center(1.1)
        elif event.keysym == 'minus':
            # 缩小
            self.zoom_at_center(0.9)
        elif event.keysym == '0':
            # 重置缩放
            self.reset_zoom()
    
    def zoom_at_center(self, scale):
        """在中心点缩放"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x = canvas_width / 2
        y = canvas_height / 2
        self.zoom_at_point(x, y, scale)
    
    def reset_zoom(self, event=None):
        """重置缩放到适合窗口大小"""
        self.scale_factor = 1.0
        self.update_image()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
    
    def update_image(self):
        """更新显示的图像"""
        # 计算新的图像尺寸
        new_width = int(self.original_image.width * self.scale_factor)
        new_height = int(self.original_image.height * self.scale_factor)
        
        # 调整图像大小
        if self.scale_factor == 1.0:
            self.current_image = self.original_image
        else:
            # 根据缩放因子选择合适的重采样方法
            if self.scale_factor > 1.0:
                resample = Image.Resampling.LANCZOS
            else:
                resample = Image.Resampling.LANCZOS
            
            self.current_image = self.original_image.resize(
                (new_width, new_height), resample
            )
        
        # 转换为PhotoImage
        self.photo = ImageTk.PhotoImage(self.current_image)
        
        # 清除旧图像
        self.canvas.delete("image")
        
        # 显示新图像
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags="image")
        
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def fit_to_window(self):
        """适应窗口大小"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # 计算适合窗口的缩放因子
        scale_x = canvas_width / self.original_image.width
        scale_y = canvas_height / self.original_image.height
        
        self.scale_factor = min(scale_x, scale_y, 1.0)  # 不放大，只缩小
        self.update_image()
        
        # 居中显示
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

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

class CoastlineExtractor:
    """海岸线提取器"""
    
    def __init__(self, model_path=None, device='cpu'):
        """
        初始化海岸线提取器
        
        参数:
        model_path: str, 训练好的模型路径
        device: str, 计算设备
        """
        self.device = device
        self.model = UNet(n_channels=3, n_classes=2)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"已加载模型: {model_path}")
        
        self.model.to(device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_coastline_from_image(self, image_path, output_dir=None, dilation_size=5):
        """
        从单张图像提取海岸线（支持TIF格式）
        
        参数:
        image_path: str, 图像路径
        output_dir: str, 输出目录
        dilation_size: int, 膨胀操作核大小
        
        返回:
        dict: 海岸线提取结果
        """
        try:
            # 读取图像（支持TIF格式）
            if image_path.lower().endswith(('.tif', '.tiff')):
                image = self.load_tif_image(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            
            original_size = image.size
            
            # 预处理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                output = self.model(input_tensor)
                pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # 调整mask尺寸到原图大小
            pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                         original_size, interpolation=cv2.INTER_NEAREST)
            
            # 提取海岸线轮廓（使用膨胀操作）
            coastlines, coastline_mask = self.extract_coastline_contours(
                pred_mask_resized, dilation_kernel_size=dilation_size
            )
            
            # 创建结果
            result = {
                'image_path': image_path,
                'image_size': original_size,
                'water_mask': pred_mask_resized,
                'coastline_mask': coastline_mask,
                'coastlines': coastlines,
                'coastline_count': len(coastlines),
                'dilation_size': dilation_size,
                'extraction_time': str(datetime.now())
            }
            
            # 保存结果
            if output_dir:
                self.save_extraction_result(result, output_dir)
            
            return result
            
        except Exception as e:
            print(f"提取海岸线时出错 {image_path}: {e}")
            return None
    
    def load_tif_image(self, tif_path):
        """
        加载TIF格式图像（与tif_to_image.py和训练程序保持一致的水体增强）
        用于模型预测
        
        参数:
        tif_path: str, TIF文件路径
        
        返回:
        PIL.Image: RGB图像
        """
        try:
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
            
        except Exception as e:
            print(f"加载TIF图像失败 {tif_path}: {e}")
            return Image.new('RGB', (512, 512), (0, 0, 0))
    
    def load_tif_image_for_display(self, tif_path):
        """
        加载TIF格式图像用于GUI显示（不应用水体增强）
        
        参数:
        tif_path: str, TIF文件路径
        
        返回:
        PIL.Image: RGB图像（原始显示效果）
        """
        try:
            dataset = gdal.Open(tif_path)
            if dataset is None:
                raise ValueError(f"无法打开TIF文件: {tif_path}")
            
            # 读取波段数据
            bands = []
            for i in range(1, min(dataset.RasterCount + 1, 7)):  # 最多读取6个波段
                band = dataset.GetRasterBand(i)
                data = band.ReadAsArray()
                bands.append(data)
            
            bands = np.array(bands)
            
            # 创建RGB图像（标准RGB组合用于自然显示）
            if bands.shape[0] >= 3:
                # 使用标准前三个波段作为RGB（通常前三个波段是RGB）
                rgb = np.dstack([bands[0], bands[1], bands[2]])  # 标准RGB顺序
            else:
                # 灰度图像
                gray = bands[0]
                rgb = np.dstack([gray, gray, gray])
            
            # 标准归一化（不应用水体增强）
            rgb_normalized = self.normalize_image_for_display(rgb)
            return Image.fromarray(rgb_normalized.astype(np.uint8))
            
        except Exception as e:
            print(f"加载TIF图像失败 {tif_path}: {e}")
            return Image.new('RGB', (512, 512), (0, 0, 0))
    
    def normalize_image_for_display(self, rgb):
        """
        方法1_标准RGB的图像归一化用于显示（不应用水体增强）
        采用简单线性拉伸方式，保持最自然的显示效果
        
        参数:
        rgb: numpy.ndarray, RGB图像数组
        
        返回:
        numpy.ndarray: 归一化后的图像
        """
        # 确保输入是3通道RGB图像
        if rgb.shape[2] < 3:
            # 灰度图像扩展为RGB
            gray = rgb[:, :, 0]
            processed_rgb = np.dstack([gray, gray, gray])
        else:
            # 使用现有的RGB数据（只取前3个通道）
            processed_rgb = rgb[:, :, :3].copy()
        
        # 简单线性拉伸（与测试程序方法1完全一致）
        normalized = np.zeros_like(processed_rgb)
        for i in range(3):  # 只处理RGB三个通道
            band = processed_rgb[:, :, i].astype(np.float64)
            
            # 计算百分位数进行拉伸
            p2, p98 = np.percentile(band, [2, 98])
            
            # 避免除零错误
            if p98 - p2 > 0:
                band_stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            else:
                band_stretched = np.clip(band, 0, 255)
            
            normalized[:, :, i] = band_stretched
        
        return normalized
    
    def enhance_image_for_water(self, rgb):
        """
        增强图像对比度，突出水体区域（与tif_to_image.py和训练程序保持一致）
        
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
    
    def extract_coastline_contours(self, water_mask, dilation_kernel_size=5):
        """
        从水体mask提取海岸线轮廓（使用膨胀操作）
        
        参数:
        water_mask: numpy.ndarray, 水体分割mask
        dilation_kernel_size: int, 膨胀操作核大小
        
        返回:
        tuple: (coastlines, dilated_mask) 海岸线轮廓点列表和膨胀后的mask
        """
        # 创建膨胀核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (dilation_kernel_size, dilation_kernel_size))
        
        # 对水体区域进行膨胀操作
        dilated_mask = cv2.dilate(water_mask, kernel, iterations=1)
        
        # 计算膨胀区域的边界（膨胀后的区域减去原始区域）
        coastline_mask = dilated_mask - water_mask
        
        # 查找海岸线轮廓
        contours, _ = cv2.findContours(coastline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        coastlines = []
        for contour in contours:
            if len(contour) > 10:  # 过滤太短的轮廓
                # 简化轮廓
                epsilon = 0.002 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                # 转换为点列表
                points = simplified.reshape(-1, 2).tolist()
                coastlines.append(points)
        
        return coastlines, coastline_mask
    
    def save_extraction_result(self, result, output_dir):
        """
        保存海岸线提取结果
        
        参数:
        result: dict, 提取结果
        output_dir: str, 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
        
        # 保存水体mask
        water_mask_path = os.path.join(output_dir, f"{base_name}_water_mask.png")
        Image.fromarray(result['water_mask'] * 255).save(water_mask_path)
        
        # 保存海岸线mask
        coastline_mask_path = os.path.join(output_dir, f"{base_name}_coastline_mask.png")
        Image.fromarray(result['coastline_mask'] * 255).save(coastline_mask_path)
        
        # 保存海岸线数据
        coastline_path = os.path.join(output_dir, f"{base_name}_coastlines.json")
        coastline_data = {
            'image_path': result['image_path'],
            'image_size': result['image_size'],
            'coastlines': result['coastlines'],
            'coastline_count': result['coastline_count'],
            'dilation_size': result.get('dilation_size', 5),
            'extraction_time': result['extraction_time']
        }
        
        with open(coastline_path, 'w', encoding='utf-8') as f:
            json.dump(coastline_data, f, indent=2, ensure_ascii=False)
        
        # 创建可视化图像
        self.create_coastsat_style_visualization(result, output_dir)
        
        print(f"结果已保存到: {output_dir}")
    
    def create_coastsat_style_visualization(self, result, output_dir):
        """
        创建CoastSat风格的海岸线可视化
        
        参数:
        result: dict, 提取结果
        output_dir: str, 输出目录
        """
        # 读取原图（用于显示，不应用水体增强）
        if result['image_path'].lower().endswith(('.tif', '.tiff')):
            image = self.load_tif_image_for_display(result['image_path'])
        else:
            image = Image.open(result['image_path'])
        
        # 创建CoastSat风格的多面板显示
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.3)
        
        # 主图：原图 + 海岸线叠加
        ax_main = fig.add_subplot(gs[:2, :2])
        ax_main.imshow(image)
        
        # 叠加海岸线（CoastSat红色风格）
        for i, coastline in enumerate(result['coastlines']):
            if len(coastline) > 2:
                coastline_array = np.array(coastline)
                ax_main.plot(coastline_array[:, 0], coastline_array[:, 1], 
                           'r-', linewidth=3, alpha=0.8, label=f'海岸线 {i+1}' if i < 3 else '')
        
        ax_main.set_title(f'海岸线检测结果\\n{os.path.basename(result["image_path"])}', 
                         fontsize=16, fontweight='bold')
        ax_main.axis('off')
        if len(result['coastlines']) <= 3:
            ax_main.legend(loc='upper right')
        
        # 水体分割结果（蓝色）
        ax_water = fig.add_subplot(gs[0, 2])
        water_colored = np.zeros((*result['water_mask'].shape, 3))
        water_colored[result['water_mask'] == 1] = [0, 0.4, 0.8]  # 深蓝色
        ax_water.imshow(water_colored)
        ax_water.set_title('水体区域\\n(蓝色)', fontsize=12, fontweight='bold')
        ax_water.axis('off')
        
        # 海岸线mask（白色）
        ax_coast = fig.add_subplot(gs[0, 3])
        ax_coast.imshow(result['coastline_mask'], cmap='gray')
        ax_coast.set_title('海岸线区域\\n(白色)', fontsize=12, fontweight='bold')
        ax_coast.axis('off')
        
        # 综合叠加图
        ax_combined = fig.add_subplot(gs[1, 2])
        combined_img = np.array(image.copy())
        
        # 调整尺寸
        display_size = combined_img.shape[:2][::-1]  # (width, height)
        water_mask_resized = cv2.resize(result['water_mask'].astype(np.uint8), 
                                      display_size, interpolation=cv2.INTER_NEAREST)
        coastline_mask_resized = cv2.resize(result['coastline_mask'].astype(np.uint8), 
                                          display_size, interpolation=cv2.INTER_NEAREST)
        
        # 水体半透明叠加
        water_coords = np.where(water_mask_resized == 1)
        if len(water_coords[0]) > 0:
            combined_img[water_coords[0], water_coords[1]] = \
                combined_img[water_coords[0], water_coords[1]] * 0.6 + np.array([0, 100, 200]) * 0.4
        
        # 海岸线白色叠加
        coastline_coords = np.where(coastline_mask_resized == 1)
        if len(coastline_coords[0]) > 0:
            combined_img[coastline_coords[0], coastline_coords[1]] = [255, 255, 255]
        
        ax_combined.imshow(combined_img.astype(np.uint8))
        ax_combined.set_title('综合结果', fontsize=12, fontweight='bold')
        ax_combined.axis('off')
        
        # 统计信息面板
        ax_stats = fig.add_subplot(gs[1, 3])
        ax_stats.axis('off')
        
        # 计算统计信息
        total_pixels = result['water_mask'].size
        water_pixels = np.sum(result['water_mask'])
        coastline_pixels = np.sum(result['coastline_mask'])
        water_ratio = water_pixels / total_pixels * 100
        
        stats_text = f"""
        📊 检测统计
        
        图像尺寸: {result['image_size'][0]} × {result['image_size'][1]}
        总像素数: {total_pixels:,}
        水体像素: {water_pixels:,}
        海岸线像素: {coastline_pixels:,}
        水体占比: {water_ratio:.1f}%
        海岸线数量: {result['coastline_count']}
        膨胀核大小: {result.get('dilation_size', 5)}
        处理时间: {result['extraction_time'][:19]}
        """
        
        ax_stats.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 海岸线长度分析
        ax_length = fig.add_subplot(gs[2, :2])
        if result['coastlines']:
            coastline_lengths = []
            for coastline in result['coastlines']:
                if len(coastline) > 1:
                    # 计算海岸线长度（像素单位）
                    points = np.array(coastline)
                    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
                    total_length = np.sum(distances)
                    coastline_lengths.append(total_length)
            
            if coastline_lengths:
                ax_length.bar(range(1, len(coastline_lengths)+1), coastline_lengths, 
                            color='coral', alpha=0.7, edgecolor='red')
                ax_length.set_xlabel('海岸线编号')
                ax_length.set_ylabel('长度 (像素)')
                ax_length.set_title('各海岸线长度分布', fontsize=12, fontweight='bold')
                ax_length.grid(True, alpha=0.3)
                
                # 添加数值标签
                for i, length in enumerate(coastline_lengths):
                    ax_length.text(i+1, length + max(coastline_lengths)*0.01, 
                                  f'{length:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 水体分布直方图
        ax_hist = fig.add_subplot(gs[2, 2:])
        
        # 创建NDWI指数用于分析（如果是多光谱图像）
        try:
            if result['image_path'].lower().endswith(('.tif', '.tiff')):
                # 对于TIF图像，尝试计算NDWI
                dataset = gdal.Open(result['image_path'])
                if dataset and dataset.RasterCount >= 4:
                    # 读取NIR和Green波段
                    nir_band = dataset.GetRasterBand(4).ReadAsArray()
                    green_band = dataset.GetRasterBand(2).ReadAsArray()
                    
                    # 计算NDWI
                    ndwi = (green_band.astype(float) - nir_band.astype(float)) / \
                           (green_band.astype(float) + nir_band.astype(float) + 1e-8)
                    
                    # 只显示水体区域的NDWI值
                    water_ndwi = ndwi[result['water_mask'] == 1]
                    other_ndwi = ndwi[result['water_mask'] == 0]
                    
                    ax_hist.hist(other_ndwi.flatten(), bins=50, alpha=0.5, color='brown', 
                               label='非水体', density=True)
                    ax_hist.hist(water_ndwi.flatten(), bins=50, alpha=0.7, color='blue', 
                               label='水体', density=True)
                    ax_hist.set_xlabel('NDWI值')
                    ax_hist.set_ylabel('密度')
                    ax_hist.set_title('水体指数(NDWI)分布', fontsize=12, fontweight='bold')
                    ax_hist.legend()
                    ax_hist.grid(True, alpha=0.3)
                else:
                    raise Exception("无法计算NDWI")
            else:
                raise Exception("非TIF格式")
                
        except:
            # 如果无法计算NDWI，显示像素强度分布
            img_array = np.array(image)
            
            # RGB通道分析
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                channel_values = img_array[:, :, i].flatten()
                ax_hist.hist(channel_values, bins=50, alpha=0.5, color=color, 
                           label=f'{color.upper()}通道', density=True)
            
            ax_hist.set_xlabel('像素值')
            ax_hist.set_ylabel('密度')
            ax_hist.set_title('RGB通道强度分布', fontsize=12, fontweight='bold')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
        
        # 添加整体标题
        fig.suptitle('🌊 CoastSat风格海岸线提取分析报告', fontsize=20, fontweight='bold', y=0.98)
        
        # 保存图像
        base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
        viz_path = os.path.join(output_dir, f"{base_name}_coastsat_analysis.png")
        plt.savefig(viz_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return viz_path

class CoastlineGUI:
    """海水区域分割与海岸线提取系统 - 现代化工业界面"""
    
    def __init__(self, root):
        """
        初始化GUI
        
        参数:
        root: tk.Tk, 主窗口
        """
        self.root = root
        self.root.title("🌊 海水区域分割与海岸线提取系统 v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # 设置现代化主题
        self.setup_styles()
        
        # 初始化变量
        self.model_path = tk.StringVar()
        self.image_paths = []  # 支持多张图片
        self.current_image_index = 0
        self.dilation_size = 20  # 默认膨胀核大小为20
        self.extractor = None
        self.current_results = []  # 存储多个结果
        self.is_batch_mode = False
        
        self.setup_ui()
        
        # 自动加载默认模型
        self.auto_load_default_model()
    
    def setup_styles(self):
        """设置现代化界面风格"""
        self.style = ttk.Style()
        
        # 配置现代化颜色主题
        self.colors = {
            'primary': '#2196F3',      # 蓝色
            'primary_dark': '#1976D2',
            'secondary': '#FF9800',    # 橙色
            'success': '#4CAF50',      # 绿色
            'warning': '#FF5722',      # 红橙色
            'info': '#00BCD4',         # 青色
            'light': '#F5F5F5',        # 浅灰
            'dark': '#424242',         # 深灰
            'white': '#FFFFFF'
        }
        
        # 设置ttk样式
        self.style.theme_use('clam')
        
        # 配置按钮样式
        self.style.configure('Primary.TButton', 
                           background=self.colors['primary'],
                           foreground='white',
                           font=('Microsoft YaHei', 10, 'bold'),
                           relief='flat',
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Success.TButton',
                           background=self.colors['success'],
                           foreground='white',
                           font=('Microsoft YaHei', 10, 'bold'),
                           relief='flat',
                           borderwidth=0)
        
        self.style.configure('Warning.TButton',
                           background=self.colors['warning'],
                           foreground='white',
                           font=('Microsoft YaHei', 10, 'bold'),
                           relief='flat',
                           borderwidth=0)
        
        # 配置标签框样式
        self.style.configure('Modern.TLabelframe',
                           background='white',
                           borderwidth=1,
                           relief='solid')
        
        self.style.configure('Modern.TLabelframe.Label',
                           background='white',
                           foreground=self.colors['primary'],
                           font=('Microsoft YaHei', 11, 'bold'))
    
    def auto_load_default_model(self):
        """自动加载默认模型"""
        default_model_path = "./models/best_water_segmentation_model.pth"
        if os.path.exists(default_model_path):
            self.model_path.set(default_model_path)
            self.load_model_silent()
    
    def load_model_silent(self):
        """静默加载模型（不弹窗提示）"""
        if not self.model_path.get():
            return
        
        try:
            self.status_var.set("正在加载模型...")
            self.root.update()
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.extractor = CoastlineExtractor(self.model_path.get(), device=device)
            
            print(f"✓ 模型加载成功: {self.model_path.get()}")
            print(f"✓ 使用设备: {device}")
            self.status_var.set(f"模型已就绪 (设备: {device}) | 膨胀核大小: {self.dilation_size}")
            self.model_status_label.config(text="✓ 已加载", foreground=self.colors['success'])
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            self.status_var.set("模型加载失败")
            self.model_status_label.config(text="❌ 失败", foreground=self.colors['warning'])

    def setup_ui(self):
        """设置现代化用户界面"""
        # 主容器
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 顶部标题栏
        self.create_header(main_container)
        
        # 控制面板区域
        self.create_control_panel(main_container)
        
        # 图像列表和预览区域
        self.create_image_panel(main_container)
        
        # 结果显示区域
        self.create_results_panel(main_container)
        
        # 底部状态栏
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        """创建顶部标题栏"""
        header_frame = tk.Frame(parent, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        # 标题
        title_label = tk.Label(header_frame, 
                              text="🌊 海水区域分割与海岸线提取系统",
                              bg=self.colors['primary'],
                              fg='white',
                              font=('Microsoft YaHei', 18, 'bold'))
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # 版本信息
        version_label = tk.Label(header_frame,
                                text="v2.0 | CoastSat Enhanced",
                                bg=self.colors['primary'],
                                fg='white',
                                font=('Microsoft YaHei', 10))
        version_label.pack(side=tk.RIGHT, padx=20, pady=20)
    
    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="🔧 控制面板", style='Modern.TLabelframe', padding=15)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 第一行：模型配置
        model_row = tk.Frame(control_frame, bg='white')
        model_row.pack(fill=tk.X, pady=5)
        
        tk.Label(model_row, text="🤖 模型:", bg='white', font=('Microsoft YaHei', 10, 'bold')).pack(side=tk.LEFT)
        model_entry = ttk.Entry(model_row, textvariable=self.model_path, width=45, font=('Consolas', 9))
        model_entry.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(model_row, text="📁 选择", command=self.select_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_row, text="⚡ 加载", command=self.load_model, style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        self.model_status_label = tk.Label(model_row, text="⏳ 未加载", bg='white', font=('Microsoft YaHei', 9))
        self.model_status_label.pack(side=tk.LEFT, padx=10)
        
        # 第二行：图像配置
        image_row = tk.Frame(control_frame, bg='white')
        image_row.pack(fill=tk.X, pady=10)
        
        tk.Label(image_row, text="🖼️ 图像:", bg='white', font=('Microsoft YaHei', 10, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(image_row, text="📂 选择单张", command=self.select_single_image, style='Success.TButton').pack(side=tk.LEFT, padx=10)
        ttk.Button(image_row, text="📁 批量选择", command=self.select_multiple_images, style='Success.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(image_row, text="🗂️ 选择文件夹", command=self.select_folder, style='Success.TButton').pack(side=tk.LEFT, padx=5)
        
        self.image_count_label = tk.Label(image_row, text="📊 已选择: 0 张", bg='white', font=('Microsoft YaHei', 9))
        self.image_count_label.pack(side=tk.LEFT, padx=20)
        
        # 第三行：处理按钮
        process_row = tk.Frame(control_frame, bg='white')
        process_row.pack(fill=tk.X, pady=10)
        
        tk.Label(process_row, text="⚙️ 操作:", bg='white', font=('Microsoft YaHei', 10, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(process_row, text="🚀 开始处理", command=self.process_images, style='Primary.TButton').pack(side=tk.LEFT, padx=10)
        ttk.Button(process_row, text="💾 保存结果", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_row, text="🗑️ 清除", command=self.clear_results, style='Warning.TButton').pack(side=tk.LEFT, padx=5)
        
        # 膨胀核大小提示
        dilation_info = tk.Label(process_row, text=f"💡 膨胀核大小: {self.dilation_size} (固定)", 
                                bg='white', font=('Microsoft YaHei', 9), fg=self.colors['info'])
        dilation_info.pack(side=tk.LEFT, padx=20)
    def create_image_panel(self, parent):
        """创建图像列表和预览面板"""
        image_frame = ttk.LabelFrame(parent, text="📷 图像管理", style='Modern.TLabelframe', padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 图像列表框
        list_frame = tk.Frame(image_frame, bg='white')
        list_frame.pack(fill=tk.X)
        
        # 列表标题
        tk.Label(list_frame, text="图像列表:", bg='white', font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W)
        
        # 创建列表框和滚动条
        list_container = tk.Frame(list_frame, bg='white')
        list_container.pack(fill=tk.X, pady=5)
        
        self.image_listbox = tk.Listbox(list_container, height=4, font=('Consolas', 9),
                                       selectmode=tk.SINGLE, activestyle='dotbox')
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # 图像操作按钮
        image_ops_frame = tk.Frame(list_frame, bg='white')
        image_ops_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(image_ops_frame, text="🔼 上移", command=self.move_image_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(image_ops_frame, text="🔽 下移", command=self.move_image_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(image_ops_frame, text="❌ 移除", command=self.remove_selected_image, style='Warning.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(image_ops_frame, text="🗑️ 清空", command=self.clear_image_list, style='Warning.TButton').pack(side=tk.LEFT, padx=2)
    
    def create_results_panel(self, parent):
        """创建结果显示面板"""
        result_frame = ttk.LabelFrame(parent, text="📊 处理结果", style='Modern.TLabelframe', padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Notebook用于切换显示
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原图标签页
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="🖼️ 原始图像")
        
        # 水体分割标签页
        self.water_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.water_frame, text="🌊 水体分割")
        
        # 海岸线标签页
        self.coastline_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.coastline_frame, text="🏖️ 海岸线")
        
        # 综合结果标签页
        self.combined_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.combined_frame, text="📈 综合结果")
        
        # 添加图像导航控制
        nav_frame = tk.Frame(result_frame, bg='white', height=40)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        nav_frame.pack_propagate(False)
        
        ttk.Button(nav_frame, text="◀ 上一张", command=self.prev_image).pack(side=tk.LEFT, padx=10)
        
        self.image_nav_label = tk.Label(nav_frame, text="0 / 0", bg='white', font=('Microsoft YaHei', 10, 'bold'))
        self.image_nav_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(nav_frame, text="下一张 ▶", command=self.next_image).pack(side=tk.LEFT, padx=10)
        
        # 图像控制按钮
        control_frame = tk.Frame(nav_frame, bg='white')
        control_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(control_frame, text="🔄 重置缩放", command=self.reset_all_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="📐 适应窗口", command=self.fit_all_to_window).pack(side=tk.LEFT, padx=2)
        
        # 使用说明
        help_label = tk.Label(nav_frame, text="💡 滚轮缩放 | 左键拖拽 | 双击重置", 
                             bg='white', font=('Microsoft YaHei', 8), fg='gray')
        help_label.pack(side=tk.RIGHT, padx=20)
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = tk.Frame(parent, bg=self.colors['light'], height=30)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="🚀 系统就绪")
        self.status_bar = tk.Label(status_frame, textvariable=self.status_var,
                                  bg=self.colors['light'], fg=self.colors['dark'],
                                  font=('Microsoft YaHei', 9), anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=10, pady=5)
    
    def select_model(self):
        """选择模型文件"""
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def select_single_image(self):
        """选择单张图像"""
        filename = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff"), ("所有文件", "*.*")]
        )
        if filename:
            self.image_paths = [filename]
            self.update_image_list()
            self.current_image_index = 0
            self.is_batch_mode = False
    
    def select_multiple_images(self):
        """批量选择多张图像"""
        filenames = filedialog.askopenfilenames(
            title="批量选择图像文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff"), ("所有文件", "*.*")]
        )
        if filenames:
            self.image_paths = list(filenames)
            self.update_image_list()
            self.current_image_index = 0
            self.is_batch_mode = len(filenames) > 1
    
    def select_folder(self):
        """选择文件夹批量导入图像"""
        folder = filedialog.askdirectory(title="选择包含图像的文件夹")
        if folder:
            # 获取文件夹中的所有图像文件
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                import glob
                image_files.extend(glob.glob(os.path.join(folder, ext)))
                image_files.extend(glob.glob(os.path.join(folder, ext.upper())))
            
            if image_files:
                self.image_paths = sorted(image_files)  # 排序
                self.update_image_list()
                self.current_image_index = 0
                self.is_batch_mode = len(image_files) > 1
            else:
                messagebox.showwarning("警告", "所选文件夹中没有找到图像文件")
    
    def update_image_list(self):
        """更新图像列表显示"""
        self.image_listbox.delete(0, tk.END)
        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path)
            self.image_listbox.insert(tk.END, f"{i+1:2d}. {filename}")
        
        self.image_count_label.config(text=f"📊 已选择: {len(self.image_paths)} 张")
        self.update_nav_label()
        
        # 如果有图像，选中第一张
        if self.image_paths:
            self.image_listbox.selection_set(0)
    
    def on_image_select(self, event):
        """图像列表选择事件"""
        selection = self.image_listbox.curselection()
        if selection:
            self.current_image_index = selection[0]
            self.update_nav_label()
            # 如果已有结果，显示对应结果
            if self.current_results and self.current_image_index < len(self.current_results):
                if self.current_results[self.current_image_index]:
                    self.display_results(self.current_results[self.current_image_index])
    
    def move_image_up(self):
        """上移选中的图像"""
        selection = self.image_listbox.curselection()
        if selection and selection[0] > 0:
            idx = selection[0]
            # 交换位置
            self.image_paths[idx], self.image_paths[idx-1] = self.image_paths[idx-1], self.image_paths[idx]
            if idx < len(self.current_results):
                self.current_results[idx], self.current_results[idx-1] = self.current_results[idx-1], self.current_results[idx]
            
            self.update_image_list()
            self.image_listbox.selection_set(idx-1)
            self.current_image_index = idx-1
    
    def move_image_down(self):
        """下移选中的图像"""
        selection = self.image_listbox.curselection()
        if selection and selection[0] < len(self.image_paths) - 1:
            idx = selection[0]
            # 交换位置
            self.image_paths[idx], self.image_paths[idx+1] = self.image_paths[idx+1], self.image_paths[idx]
            if idx < len(self.current_results) - 1:
                self.current_results[idx], self.current_results[idx+1] = self.current_results[idx+1], self.current_results[idx]
            
            self.update_image_list()
            self.image_listbox.selection_set(idx+1)
            self.current_image_index = idx+1
    
    def remove_selected_image(self):
        """移除选中的图像"""
        selection = self.image_listbox.curselection()
        if selection:
            idx = selection[0]
            del self.image_paths[idx]
            if idx < len(self.current_results):
                del self.current_results[idx]
            
            self.update_image_list()
            # 调整当前索引
            if self.current_image_index >= len(self.image_paths):
                self.current_image_index = max(0, len(self.image_paths) - 1)
            
            if self.image_paths and self.current_image_index < len(self.image_paths):
                self.image_listbox.selection_set(self.current_image_index)
    
    def clear_image_list(self):
        """清空图像列表"""
        self.image_paths = []
        self.current_results = []
        self.current_image_index = 0
        self.update_image_list()
        self.clear_results()
    
    def prev_image(self):
        """上一张图像"""
        if self.image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_image_index)
            self.update_nav_label()
            if self.current_results and self.current_image_index < len(self.current_results):
                if self.current_results[self.current_image_index]:
                    self.display_results(self.current_results[self.current_image_index])
    
    def next_image(self):
        """下一张图像"""
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_image_index)
            self.update_nav_label()
            if self.current_results and self.current_image_index < len(self.current_results):
                if self.current_results[self.current_image_index]:
                    self.display_results(self.current_results[self.current_image_index])
    
    def update_nav_label(self):
        """更新导航标签"""
        if self.image_paths:
            self.image_nav_label.config(text=f"{self.current_image_index + 1} / {len(self.image_paths)}")
        else:
            self.image_nav_label.config(text="0 / 0")
    
    def load_model(self):
        """加载模型（带弹窗提示）"""
        if not self.model_path.get():
            messagebox.showerror("错误", "请先选择模型文件")
            return
        
        try:
            self.status_var.set("正在加载模型...")
            self.root.update()
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.extractor = CoastlineExtractor(self.model_path.get(), device=device)
            
            self.status_var.set(f"模型已就绪 (设备: {device}) | 膨胀核大小: {self.dilation_size}")
            self.model_status_label.config(text="✓ 已加载", foreground=self.colors['success'])
            print(f"✓ 模型加载成功: {self.model_path.get()}")
            print(f"✓ 使用设备: {device}")
            messagebox.showinfo("成功", f"模型加载成功！\n设备: {device}")
            
        except Exception as e:
            self.status_var.set("模型加载失败")
            self.model_status_label.config(text="❌ 失败", foreground=self.colors['warning'])
            print(f"❌ 模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
    
    def process_images(self):
        """处理图像（支持单张和批量）"""
        if not self.extractor:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        if not self.image_paths:
            messagebox.showerror("错误", "请先选择图像文件")
            return
        
        def process_thread():
            try:
                self.current_results = []
                total_images = len(self.image_paths)
                
                for i, image_path in enumerate(self.image_paths):
                    # 更新状态
                    progress_text = f"正在处理图像 {i+1}/{total_images}: {os.path.basename(image_path)}"
                    self.status_var.set(progress_text)
                    self.root.update()
                    
                    print(f"\n🔄 {progress_text}")
                    
                    # 提取海岸线
                    result = self.extractor.extract_coastline_from_image(
                        image_path,
                        dilation_size=self.dilation_size
                    )
                    
                    self.current_results.append(result)
                    
                    if result:
                        print(f"✓ 完成 - 找到 {result['coastline_count']} 条海岸线")
                        
                        # 如果是当前显示的图像，立即更新显示
                        if i == self.current_image_index:
                            self.display_results(result)
                    else:
                        print("❌ 处理失败")
                
                # 处理完成
                successful_count = sum(1 for r in self.current_results if r is not None)
                
                if self.is_batch_mode:
                    self.status_var.set(f"✅ 批量处理完成 - 成功: {successful_count}/{total_images}")
                    print(f"\n🎉 批量处理完成！成功处理 {successful_count}/{total_images} 张图像")
                else:
                    if successful_count > 0:
                        result = self.current_results[0]
                        self.status_var.set(f"✅ 处理完成 - 找到 {result['coastline_count']} 条海岸线")
                        print(f"🎉 处理完成！找到 {result['coastline_count']} 条海岸线")
                    else:
                        self.status_var.set("❌ 处理失败")
                        print("❌ 处理失败")
                
                # 显示当前图像的结果
                if (self.current_results and self.current_image_index < len(self.current_results) 
                    and self.current_results[self.current_image_index]):
                    self.display_results(self.current_results[self.current_image_index])
                
            except Exception as e:
                self.status_var.set(f"❌ 处理出错: {str(e)}")
                print(f"❌ 处理出错: {str(e)}")
                messagebox.showerror("错误", f"处理出错: {str(e)}")
        
        # 在后台线程中处理
        import threading
        threading.Thread(target=process_thread, daemon=True).start()
    
    def display_results(self, result):
        """显示处理结果"""
        try:
            # 加载原图（用于显示，不应用水体增强）
            if result['image_path'].lower().endswith(('.tif', '.tiff')):
                original_image = self.extractor.load_tif_image_for_display(result['image_path'])
            else:
                original_image = Image.open(result['image_path'])
            
            # 调整图像大小以适应显示
            max_size = (600, 400)
            original_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 显示原图
            self.display_image_in_frame(original_image, self.original_frame, "原始图像")
            
            # 创建水体分割显示（蓝色）
            water_display = self.create_water_display(result, original_image.size)
            self.display_image_in_frame(water_display, self.water_frame, "水体区域（蓝色）")
            
            # 创建海岸线显示（白色）
            coastline_display = self.create_coastline_display(result, original_image.size)
            self.display_image_in_frame(coastline_display, self.coastline_frame, "海岸线（白色）")
            
            # 创建综合显示
            combined_display = self.create_combined_display(result, original_image)
            self.display_image_in_frame(combined_display, self.combined_frame, 
                                      f"综合结果\\n水体（蓝色）+ 海岸线（白色）")
            
        except Exception as e:
            messagebox.showerror("显示错误", f"显示结果时出错: {str(e)}")
    
    def create_water_display(self, result, display_size):
        """创建水体分割显示图像"""
        # 调整mask尺寸
        water_mask = cv2.resize(result['water_mask'], display_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建蓝色水体图像
        water_image = np.zeros((*water_mask.shape, 3), dtype=np.uint8)
        water_image[water_mask == 1] = [0, 0, 255]  # 蓝色
        
        return Image.fromarray(water_image)
    
    def create_coastline_display(self, result, display_size):
        """创建海岸线显示图像"""
        # 调整mask尺寸
        coastline_mask = cv2.resize(result['coastline_mask'], display_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建白色海岸线图像
        coastline_image = np.zeros((*coastline_mask.shape, 3), dtype=np.uint8)
        coastline_image[coastline_mask == 1] = [255, 255, 255]  # 白色
        
        return Image.fromarray(coastline_image)
    
    def create_combined_display(self, result, original_image):
        """创建综合显示图像"""
        # 转换为numpy数组
        combined = np.array(original_image.copy())
        
        # 调整mask尺寸
        display_size = original_image.size
        water_mask = cv2.resize(result['water_mask'], display_size, interpolation=cv2.INTER_NEAREST)
        coastline_mask = cv2.resize(result['coastline_mask'], display_size, interpolation=cv2.INTER_NEAREST)
        
        # 叠加水体区域（蓝色半透明）
        water_coords = np.where(water_mask == 1)
        combined[water_coords[0], water_coords[1]] = combined[water_coords[0], water_coords[1]] * 0.7 + np.array([0, 0, 255]) * 0.3
        
        # 叠加海岸线（白色）
        coastline_coords = np.where(coastline_mask == 1)
        combined[coastline_coords[0], coastline_coords[1]] = [255, 255, 255]
        
        return Image.fromarray(combined.astype(np.uint8))
    
    def display_image_in_frame(self, image, frame, title):
        """在指定框架中显示图像 - 支持缩放和拖拽"""
        # 清除框架中的旧内容
        for widget in frame.winfo_children():
            widget.destroy()
        
        # 创建可缩放拖拽的图像查看器
        image_viewer = ZoomableImageCanvas(frame, image)
        image_viewer.pack(fill=tk.BOTH, expand=True)
    
    def save_results(self):
        """保存结果（支持批量保存）"""
        if not self.current_results or not any(self.current_results):
            messagebox.showerror("错误", "没有可保存的结果")
            return
        
        output_dir = filedialog.askdirectory(title="选择保存目录")
        if output_dir:
            try:
                saved_count = 0
                for i, result in enumerate(self.current_results):
                    if result:
                        # 为每个结果创建子目录
                        image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                        result_dir = os.path.join(output_dir, f"{i+1:03d}_{image_name}")
                        
                        self.extractor.save_extraction_result(result, result_dir)
                        saved_count += 1
                        print(f"✓ 已保存: {result_dir}")
                
                if self.is_batch_mode:
                    messagebox.showinfo("成功", f"批量保存完成！\n成功保存 {saved_count} 个结果到: {output_dir}")
                else:
                    messagebox.showinfo("成功", f"结果已保存到: {output_dir}")
                    
                print(f"🎉 保存完成！共保存 {saved_count} 个结果")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                print(f"❌ 保存失败: {str(e)}")
    
    def clear_results(self):
        """清除结果"""
        self.current_results = []
        
        # 清除所有显示框架
        for frame in [self.original_frame, self.water_frame, self.coastline_frame, self.combined_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        self.status_var.set("🗑️ 已清除结果")
        print("🗑️ 已清除所有结果")
    
    def reset_all_zoom(self):
        """重置所有图像的缩放"""
        for frame in [self.original_frame, self.water_frame, self.coastline_frame, self.combined_frame]:
            for widget in frame.winfo_children():
                if isinstance(widget, ZoomableImageCanvas):
                    widget.reset_zoom()
    
    def fit_all_to_window(self):
        """让所有图像适应窗口大小"""
        for frame in [self.original_frame, self.water_frame, self.coastline_frame, self.combined_frame]:
            for widget in frame.winfo_children():
                if isinstance(widget, ZoomableImageCanvas):
                    widget.fit_to_window()
            for widget in frame.winfo_children():
                if isinstance(widget, ZoomableImageCanvas):
                    widget.fit_to_window()

def main():
    """主函数 - 默认启动图形界面"""
    print("=" * 60)
    print("🌊 海水区域分割与海岸线提取系统 v2.0")
    print("CoastSat Enhanced Edition")
    print("=" * 60)
    
    try:
        # 直接启动图形界面
        print("🚀 启动图形界面...")
        
        try:
            import tkinter as tk
            root = tk.Tk()
            app = CoastlineGUI(root)
            
            print("✅ 图形界面启动成功！")
            print("💡 提示：")
            print("  - 默认膨胀核大小已设为 20")
            print("  - 支持单张和批量图像处理")
            print("  - 会自动尝试加载默认模型")
            print("  - 界面已优化为现代工业风格")
            print("=" * 60)
            
            root.mainloop()
            
        except ImportError:
            print("❌ GUI启动失败：缺少tkinter库")
            print("请安装tkinter: pip install tk")
            
            # 提供命令行替代选项
            print("\n🔄 转为命令行模式...")
            command_line_interface()
            
        except Exception as e:
            print(f"❌ GUI启动失败: {e}")
            print("\n� 转为命令行模式...")
            command_line_interface()
    
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")

def command_line_interface():
    """命令行界面（备用选项）"""
    print("\n" + "=" * 50)
    print("📋 命令行模式")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 单张图像处理")
        print("2. 批量处理")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            # 单张图像处理
            print("\n=== 单张图像处理 ===")
            model_path = input("请输入模型文件路径 (默认: ./models/best_water_segmentation_model.pth): ").strip()
            if not model_path:
                model_path = "./models/best_water_segmentation_model.pth"
            
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                continue
            
            image_path = input("请输入图像文件路径: ").strip()
            if os.path.exists(image_path):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"使用设备: {device}")
                
                extractor = CoastlineExtractor(model_path=model_path, device=device)
                
                print("正在处理图像...")
                result = extractor.extract_coastline_from_image(
                    image_path,
                    output_dir="./coastline_results",
                    dilation_size=20  # 默认膨胀核大小
                )
                
                if result:
                    print(f"\n=== 处理完成 ===")
                    print(f"找到 {result['coastline_count']} 条海岸线")
                    print(f"图像尺寸: {result['image_size']}")
                    print(f"膨胀核大小: {result['dilation_size']}")
                    print(f"结果保存在: ./coastline_results")
                else:
                    print("❌ 图像处理失败")
            else:
                print("❌ 图像文件不存在")
        
        elif choice == '2':
            # 批量处理
            print("\n=== 批量处理功能 ===")
            model_path = input("请输入模型文件路径 (默认: ./models/best_water_segmentation_model.pth): ").strip()
            if not model_path:
                model_path = "./models/best_water_segmentation_model.pth"
            
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                continue
            
            images_dir = input("请输入图像目录路径: ").strip()
            if not os.path.exists(images_dir):
                print("❌ 目录不存在")
                continue
            
            output_dir = input("请输入输出目录路径 (默认: ./batch_results): ").strip()
            if not output_dir:
                output_dir = "./batch_results"
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            extractor = CoastlineExtractor(model_path=model_path, device=device)
            
            # 获取所有图像文件
            import glob
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                image_files.extend(glob.glob(os.path.join(images_dir, ext)))
                image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
            
            print(f"\n找到 {len(image_files)} 个图像文件")
            
            for i, image_path in enumerate(image_files, 1):
                print(f"\n处理 {i}/{len(image_files)}: {os.path.basename(image_path)}")
                
                result = extractor.extract_coastline_from_image(
                    image_path,
                    output_dir=output_dir,
                    dilation_size=20  # 默认膨胀核大小
                )
                
                if result:
                    print(f"  -> 找到 {result['coastline_count']} 条海岸线")
                else:
                    print("  -> 处理失败")
            
            print(f"\n🎉 批量处理完成！结果保存在: {output_dir}")
        
        elif choice == '3':
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()
