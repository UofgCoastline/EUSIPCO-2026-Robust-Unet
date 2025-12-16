Update from 12/16/2025 (Added PY file for extended baseline comparison - Extended_Baseline_Comparison.py) : The open-source repository has been extended to include additional baseline models for more comprehensive comparison. 
Specifically, recent water-specific segmentation methods, including WaterNet, MSWNet, HRNet-Water, and SegFormer-Lite, are incorporated under a unified training and evaluation protocol. 
This extension is intended to enable fair and transparent benchmarking, while isolating the effect of the proposed robust training constraints from architectural differences. 
All baseline methods are evaluated using identical data splits, input settings, and evaluation metrics.

Quantitative Performance Comparison
Model	IoU (mean ± std)	F1 (mean ± std)	Acc (mean ± std)
Robust U-Net	0.9645 ± 0.003	0.9819 ± 0.002	0.9810 ± 0.002
DeepLabV3+	0.9639 ± 0.005	0.9816 ± 0.003	0.9806 ± 0.003
YOLO-SEG	0.9407 ± 0.076	0.9676 ± 0.046	0.9684 ± 0.040
WaterNet(New)	0.9647 ± 0.005	0.9820 ± 0.003	0.9811 ± 0.003
MSWNet(New)	0.9658 ± 0.004	0.9826 ± 0.002	0.9817 ± 0.002
HRNet-Water(New)	0.9471 ± 0.050	0.9721 ± 0.029	0.9717 ± 0.026
SegFormer-Lite(New)	0.9625 ± 0.004	0.9809 ± 0.002	0.9799 ± 0.002

| Model            | IoU (mean ± std)   | F1 (mean ± std)    | Acc (mean ± std)   |
| ---------------- | ------------------ | ------------------ | ------------------ |
| **Robust U-Net** | **0.9645 ± 0.003** | **0.9819 ± 0.002** | **0.9810 ± 0.002** |
| DeepLabV3+       | 0.9639 ± 0.005     | 0.9816 ± 0.003     | 0.9806 ± 0.003     |
| YOLO-SEG         | 0.9407 ± 0.076     | 0.9676 ± 0.046     | 0.9684 ± 0.040     |
| WaterNet         | 0.9647 ± 0.005     | 0.9820 ± 0.003     | 0.9811 ± 0.003     |
| MSWNet           | **0.9658 ± 0.004** | **0.9826 ± 0.002** | **0.9817 ± 0.002** |
| HRNet-Water      | 0.9471 ± 0.050     | 0.9721 ± 0.029     | 0.9717 ± 0.026     |
| SegFormer-Lite   | 0.9625 ± 0.004     | 0.9809 ± 0.002     | 0.9799 ± 0.002     |


Although Robust U-Net does not achieve the absolute highest mean scores among all methods, it consistently delivers competitive performance across all metrics with markedly lower variance. This indicates improved training stability and robustness, which are critical in coastal water segmentation where spectral conditions and shoreline geometry vary significantly. In contrast, several water-specific models achieve marginally higher mean accuracy at the cost of increased model complexity or higher sensitivity to training conditions. The proposed approach therefore offers a favorable trade-off between accuracy, stability, and robustness, making it well suited for reliable large-scale coastal monitoring.

Experiments were performed on a Windows 10 Professional (64-bit) workstation with an Intel 12th Gen Core i7-12700KF processor (3.60 GHz) and 32 GB of system memory.

ICASSP-2026 Robust U-Net for Coastal Water Segmentation

Coastal water segmentation with a strong, practical baseline comparing Robust U-Net, DeepLabV3+, and YOLO-SEG on coastal satellite imagery with Labelme annotations.
This repo includes a clean PyTorch re-implementation of the models, training/evaluation loops, and plotting scripts for curves and final comparisons.

📄 Associated paper draft: “Multi-Modal Robust Enhancement for Coastal Water Segmentation: A Systematic HSV-Guided Framework” (ICASSP 2026 submission)

------------------------------------------------------
Coastal erosion is a critical global challenge, intensified by climate change, sea-level rise, and increasingly frequent extreme weather events. Shoreline retreat directly threatens infrastructure, local communities, and ecosystems, leading to loss of land, damage to property, and long-term socio-economic risks.

In Scotland and across Europe, many coastal regions are already experiencing accelerated erosion. Traditional monitoring methods — such as manual field surveys or spectral index–based remote sensing — often struggle to provide the accuracy, robustness, and scalability needed for effective decision-making.

The TERRA Project (Horizon Europe) addresses this challenge by developing digital twins for coastal zones, enabling continuous monitoring, simulation, and prediction of coastal changes. Within TERRA, our team at the University of Glasgow leads the CoastDT module, focusing on coastal water and shoreline segmentation from satellite imagery (Sentinel-2, Landsat, PlanetScope, etc.).

Our Robust U-Net framework is designed to:

Capture complex shoreline geometries with high accuracy.

Remain stable across diverse environmental and seasonal conditions.

Provide reliable inputs for erosion risk modeling and long-term digital twin simulations.

By advancing automated and physics-informed segmentation, this work contributes to practical coastal erosion monitoring, supporting local councils, environmental agencies, and policymakers in planning mitigation and adaptation strategies.








-----------------------------------------------------
Features

Robust U-Net with Residual blocks, Channel & Spatial Attention, Dilated bottleneck, Attention gates

Baselines: DeepLabV3+, YOLO-style segmentation head

Labelme JSON → binary water mask pipeline

Training/validation loops with IoU/F1/Accuracy

Curves and bar plots saved to disk

---------------------------------------------------
conda create -n coast python=3.10 -y
conda activate coast
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
pip install numpy pillow scikit-learn matplotlib

---------------------------------------------------
Quick Start

Run all models (Robust U-Net, DeepLabV3+, YOLO-SEG) end-to-end:

python main.py

----------------------------------------------------
Outputs:

training_curves.png — train/val loss, IoU、F1 Curves

coastal_comparison.png — IoU/F1/Acc 

Console summary with per-model params & inference timing

----------------------------------------------------
How to Cite

@misc{tian2025robustcoast_code,
  author       = {Zhen Tian and Christos Anagnostopoulos and Qiyuan Wang and Zhiwei Gao},
  title        = {{Robust U-Net for Coastal Water Segmentation: HSV-Guided Framework (Code Repository)}},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/UofgCoastline/ICASSP-2026-Robust-Unet}},
  note         = {Code and data for the ICASSP 2026 paper}
}




