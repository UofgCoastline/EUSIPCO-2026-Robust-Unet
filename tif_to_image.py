#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF图像预处理工具 - 为labelme标注准备图像
将TIF文件转换为适合labelme标注的PNG格式

作者: CoastSat海水标注助手
创建日期: 2025-01-26
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal
import json
from datetime import datetime

class TIFToImageConverter:
    """TIF文件转图像转换器"""
    
    def __init__(self, input_dir="./data", output_dir="./labelme_images"):
        """
        初始化转换器
        
        参数:
        input_dir: str, 输入TIF文件目录
        output_dir: str, 输出图像目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "converted"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
    
    def convert_tif_to_png(self, tif_path, enhance_water=True):
        """
        将TIF文件转换为PNG图像
        
        参数:
        tif_path: str, TIF文件路径
        enhance_water: bool, 是否增强水体对比度
        
        返回:
        tuple: (png_path, metadata)
        """
        try:
            # 打开TIF文件
            dataset = gdal.Open(tif_path)
            if dataset is None:
                print(f"无法打开文件: {tif_path}")
                return None, None
            
            # 获取文件信息
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            bands_count = dataset.RasterCount
            
            print(f"处理文件: {os.path.basename(tif_path)}")
            print(f"  尺寸: {width}x{height}")
            print(f"  波段数: {bands_count}")
            
            # 读取波段数据
            bands = []
            for i in range(1, min(bands_count + 1, 7)):  # 最多读取6个波段
                band = dataset.GetRasterBand(i)
                data = band.ReadAsArray()
                bands.append(data)
            
            bands = np.array(bands)
            
            # 创建RGB图像用于显示
            if bands.shape[0] >= 3:
                # 选择合适的波段组合来突出水体
                if enhance_water and bands.shape[0] >= 4:
                    # 使用近红外、红、绿波段组合来突出水体
                    # 一般Landsat波段: 1-Coastal, 2-Blue, 3-Green, 4-Red, 5-NIR, 6-SWIR1
                    try:
                        rgb = np.dstack([bands[4], bands[3], bands[2]])  # NIR, Red, Green
                        enhancement_type = "NIR-Red-Green (水体增强)"
                    except IndexError:
                        rgb = np.dstack([bands[2], bands[1], bands[0]])  # 标准RGB
                        enhancement_type = "标准RGB"
                else:
                    # 标准RGB组合
                    rgb = np.dstack([bands[2], bands[1], bands[0]])  # Red, Green, Blue
                    enhancement_type = "标准RGB"
            else:
                # 灰度图像
                gray = bands[0]
                rgb = np.dstack([gray, gray, gray])
                enhancement_type = "灰度"
            
            # 图像增强和归一化
            rgb_enhanced = self.enhance_image(rgb, enhance_water)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(rgb_enhanced.astype(np.uint8))
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(tif_path))[0]
            png_path = os.path.join(self.output_dir, "converted", f"{base_name}.png")
            
            # 保存PNG图像
            pil_image.save(png_path, "PNG")
            
            # 创建元数据
            metadata = {
                "original_file": tif_path,
                "png_file": png_path,
                "image_size": [width, height],
                "bands_count": bands_count,
                "enhancement_type": enhancement_type,
                "conversion_time": str(datetime.now()),
                "geo_transform": dataset.GetGeoTransform() if dataset.GetGeoTransform() else None,
                "projection": dataset.GetProjection() if dataset.GetProjection() else None
            }
            
            # 保存元数据
            metadata_path = os.path.join(self.output_dir, "metadata", f"{base_name}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"  转换完成: {png_path}")
            print(f"  增强方式: {enhancement_type}")
            
            return png_path, metadata
            
        except Exception as e:
            print(f"转换文件时出错 {tif_path}: {e}")
            return None, None
    
    def enhance_image(self, rgb, enhance_water=True):
        """
        增强图像对比度，突出水体区域
        
        参数:
        rgb: numpy.ndarray, RGB图像数组
        enhance_water: bool, 是否增强水体对比度
        
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
            
            if enhance_water and i < 3:
                # 对水体区域进行额外增强
                # 水体通常在近红外波段反射率低
                if i == 0:  # 假设第一个波段是近红外或红色
                    # 增强低值区域（可能是水体）
                    mask = band_stretched < 100
                    band_stretched[mask] = band_stretched[mask] * 0.7  # 降低亮度突出水体
            
            enhanced[:, :, i] = band_stretched
        
        return enhanced
    
    def batch_convert(self, max_files=None):
        """
        批量转换TIF文件
        
        参数:
        max_files: int, 最大转换文件数（None表示全部转换）
        
        返回:
        list: 转换结果列表
        """
        print("=== 开始批量转换TIF文件 ===")
        
        # 扫描所有TIF文件
        tif_files = []
        for year in range(2017, 2026):
            year_dir = os.path.join(self.input_dir, str(year))
            if os.path.exists(year_dir):
                for file in os.listdir(year_dir):
                    if file.lower().endswith('.tif'):
                        tif_files.append(os.path.join(year_dir, file))
        
        print(f"发现 {len(tif_files)} 个TIF文件")
        
        if max_files:
            tif_files = tif_files[:max_files]
            print(f"限制转换数量为: {max_files}")
        
        # 转换文件
        converted_files = []
        for i, tif_file in enumerate(tif_files):
            print(f"\\n进度: {i+1}/{len(tif_files)}")
            
            png_path, metadata = self.convert_tif_to_png(tif_file, enhance_water=True)
            
            if png_path and metadata:
                converted_files.append({
                    'tif_file': tif_file,
                    'png_file': png_path,
                    'metadata': metadata
                })
        
        # 创建转换汇总
        summary = {
            "total_files": len(tif_files),
            "converted_files": len(converted_files),
            "conversion_time": str(datetime.now()),
            "files": converted_files
        }
        
        summary_path = os.path.join(self.output_dir, "conversion_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\\n=== 转换完成 ===")
        print(f"转换文件数: {len(converted_files)}")
        print(f"输出目录: {self.output_dir}/converted/")
        print(f"汇总文件: {summary_path}")
        
        return converted_files
    
    def preview_conversion(self, tif_file):
        """
        预览转换效果
        
        参数:
        tif_file: str, TIF文件路径
        """
        print(f"\\n=== 预览转换: {os.path.basename(tif_file)} ===")
        
        # 转换文件
        png_path, metadata = self.convert_tif_to_png(tif_file)
        
        if not png_path:
            print("转换失败")
            return
        
        # 显示对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始TIF（简化显示）
        dataset = gdal.Open(tif_file)
        band1 = dataset.GetRasterBand(1).ReadAsArray()
        ax1.imshow(band1, cmap='gray')
        ax1.set_title("原始TIF (第1波段)")
        ax1.axis('off')
        
        # 转换后的PNG
        png_image = Image.open(png_path)
        ax2.imshow(png_image)
        ax2.set_title(f"转换后PNG\\n({metadata['enhancement_type']})")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"转换文件: {png_path}")
        print(f"图像尺寸: {metadata['image_size']}")
        print(f"增强方式: {metadata['enhancement_type']}")

def main():
    """主函数"""
    converter = TIFToImageConverter()
    
    print("=== CoastSat TIF转图像工具 ===")
    print("用于将TIF文件转换为适合labelme标注的PNG图像")
    print("转换时会增强水体对比度，便于海水区域标注")
    
    print("\\n选择操作:")
    print("1. 预览单个文件转换效果")
    print("2. 转换前10个文件（测试）")
    print("3. 转换所有文件")
    print("4. 自定义转换数量")
    
    try:
        choice = input("\\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            # 预览转换
            tif_files = []
            for year in range(2017, 2026):
                year_dir = os.path.join("./data", str(year))
                if os.path.exists(year_dir):
                    for file in os.listdir(year_dir):
                        if file.lower().endswith('.tif'):
                            tif_files.append(os.path.join(year_dir, file))
            
            if tif_files:
                converter.preview_conversion(tif_files[0])
            else:
                print("未找到TIF文件")
        
        elif choice == '2':
            # 转换前10个文件
            converter.batch_convert(max_files=10)
        
        elif choice == '3':
            # 转换所有文件
            converter.batch_convert()
        
        elif choice == '4':
            # 自定义数量
            try:
                count = int(input("请输入要转换的文件数量: "))
                converter.batch_convert(max_files=count)
            except ValueError:
                print("请输入有效数字")
        
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\\n\\n转换已取消")

if __name__ == "__main__":
    main()
