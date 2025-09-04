ICASSP-2026 Robust U-Net for Coastal Water Segmentation

Coastal water segmentation with a strong, practical baseline comparing Robust U-Net, DeepLabV3+, and YOLO-SEG on coastal satellite imagery with Labelme annotations.
This repo includes a clean PyTorch re-implementation of the models, training/evaluation loops, and plotting scripts for curves and final comparisons.

📄 Associated paper draft: “Multi-Modal Robust Enhancement for Coastal Water Segmentation: A Systematic HSV-Guided Framework” (ICASSP 2026 submission)

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # 或 CPU 版本
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




