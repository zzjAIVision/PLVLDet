# PLVLDet: A Power Line Vision-Language Detection Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"PLVLDet: A Power Line Vision-Language Detection Model with Hierarchical Cross-Modal Fusion for Transmission Line Component Recognition"**.

## Overview

PLVLDet is a vision-language detection model designed for transmission line component recognition. It achieves **38.5% mAP** in zero-shot detection (11× higher than Grounding DINO-B) and **80.8% mAP** after fine-tuning.

### Key Components

- **PowerBERT**: Domain-adapted text encoder for power line terminology via Masked Language Modeling
- **YOLOv12 Backbone**: Visual feature extractor with R-ELAN and Area Attention for elongated objects
- **PA-VL-PAN**: Power-Aware Vision-Language Path Aggregation Network with:
  - PT-CSPLayer (Text-guided Cross-Attention)
  - Strip Pooling Attention (SPA) for directional context
  - Small Object Enhancement Module (SOEM)
- **Region-Text Contrastive Learning**: Cross-modal alignment for open-vocabulary detection

### Features

- ✅ Zero-shot detection for unseen power line defects
- ✅ Hierarchical cross-modal fusion for fine-grained recognition
- ✅ Efficient architecture: 18.6M parameters, 45.2 GFLOPs
- ✅ Support for 32 categories (7 parent + 14 seen leaf + 11 unseen leaf)
- ✅ Multi-GPU distributed training support
- ✅ Comprehensive ablation study scripts

## Project Structure

```
PLVLDet/
├── configs/                    # Configuration files
│   ├── categories.yaml         # Category definitions with hierarchy
│   ├── pretrain_config.yaml    # Vision-language pretraining config
│   ├── finetune_config.yaml    # Fine-tuning config
│   ├── powerbert_pretrain.yaml # PowerBERT MLM config
│   └── ablation_config.yaml    # Ablation study config
├── models/                     # Model implementations
│   ├── powerbert.py            # Domain-adapted BERT text encoder
│   ├── backbone.py             # YOLOv12 backbone with R-ELAN
│   ├── pa_vl_pan.py            # PA-VL-PAN neck
│   ├── detector.py             # Full PLVLDet model
│   └── components.py           # Shared components
├── data/                       # Data loading and augmentation
│   ├── dataset.py              # Dataset classes
│   ├── dataloader.py           # DataLoader utilities
│   └── transforms.py           # Data augmentations
├── losses/                     # Loss functions
│   └── __init__.py             # Contrastive, CIoU, DFL losses
├── utils/                      # Utility functions
│   ├── metrics.py              # mAP calculation
│   ├── postprocess.py          # NMS and box processing
│   └── visualization.py        # Detection visualization
├── tools/                      # Data preprocessing tools
│   ├── labelme_converter.py    # LabelMe to COCO/YOLO conversion
│   ├── patch_generator.py      # 1000×1000 patch generation
│   └── dataset_splitter.py     # Route-level dataset splitting
├── scripts/                    # Training and evaluation scripts
│   ├── pretrain_powerbert.py   # Stage 1: PowerBERT MLM
│   ├── pretrain_plvldet.py     # Stage 2: VL pretraining
│   ├── finetune_plvldet.py     # Stage 3: Fine-tuning
│   ├── evaluate.py             # Model evaluation
│   ├── inference.py            # Single/batch inference
│   ├── demo.py                 # Interactive demo
│   └── ablation/               # Ablation study scripts
│       ├── run_all_ablations.py
│       ├── text_encoder_ablation.py
│       ├── pavlpan_ablation.py
│       ├── backbone_ablation.py
│       ├── freeze_strategy_ablation.py
│       ├── pretraining_ablation.py
│       └── hierarchical_label_ablation.py
└── requirements.txt            # Dependencies
```

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/xxx/PLVLDet.git
cd PLVLDet

# Create conda environment
conda create -n plvldet python=3.8 -y
conda activate plvldet

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Input Directory Structure

Place your raw data in the following structure:

```
data_images/           # High-resolution images (8000×6000 pixels)
├── route_001/
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── route_002/
└── ...

data_labels/           # LabelMe JSON annotations
├── route_001/
│   ├── IMG_0001.json
│   ├── IMG_0002.json
│   └── ...
├── route_002/
└── ...
```

### Data Processing Pipeline

```bash
# Step 1: Convert LabelMe annotations to COCO format
python tools/labelme_converter.py \
    --image_dir data_images \
    --label_dir data_labels \
    --output_dir data/annotations \
    --format coco

# Step 2: Generate 1000×1000 patches with stride 800
python tools/patch_generator.py \
    --image_dir data_images \
    --annotation_file data/annotations/annotations.json \
    --output_dir data/patches \
    --patch_size 1000 \
    --stride 800 \
    --overlap_threshold 0.1 \
    --area_ratio_threshold 0.9

# Step 3: Route-level dataset splitting (MixPLOD + PLEval)
python tools/dataset_splitter.py \
    --patch_dir data/patches \
    --output_dir data/splits \
    --pretrain_ratio 0.75 \
    --val_ratio 0.1 \
    --test_ratio 0.15
```

After processing, the dataset structure:
```
data/splits/
├── mixplod_pretrain/       # 142 routes, 361,434 patches, 21 seen categories
│   ├── images/
│   └── annotations.json
├── pleval_train/           # 33 routes, 84,335 patches, 32 categories
├── pleval_val/             # 7 routes, 18,394 patches
└── pleval_test/            # 7 routes, 17,749 patches
```

## Training

### Three-Stage Training Pipeline

#### Stage 1: PowerBERT Domain Adaptation (MLM)

Pre-train BERT on power system terminology:

```bash
python scripts/pretrain_powerbert.py \
    --corpus_path data/power_corpus.txt \
    --output_dir checkpoints/powerbert \
    --epochs 30 \
    --batch_size 32 \
    --lr 2e-5
```

If no corpus is available, the script generates synthetic training data from power domain vocabulary.

#### Stage 2: Vision-Language Pretraining on MixPLOD

Train PLVLDet with region-text contrastive learning:

```bash
# Single GPU
python scripts/pretrain_plvldet.py \
    --config configs/pretrain_config.yaml

# Multi-GPU (recommended: 4× A100 GPUs)
torchrun --nproc_per_node=4 scripts/pretrain_plvldet.py \
    --config configs/pretrain_config.yaml
```

Training settings:
- 100 epochs, batch size 32 (8 per GPU)
- Learning rate: 0.0002 → 0.0001 (cosine decay)
- Text encoder frozen
- Mosaic augmentation (p=0.5), MixUp (p=0.1)

#### Stage 3: Fine-tuning on PLEval

Fine-tune on the evaluation dataset with all 32 categories:

```bash
torchrun --nproc_per_node=4 scripts/finetune_plvldet.py \
    --config configs/finetune_config.yaml \
    --pretrained checkpoints/pretrain_best.pth
```

Training settings:
- 50 epochs, batch size 32
- Learning rate: 0.0001 (lower than pretraining)
- Lighter augmentation: Mosaic (p=0.3), MixUp (p=0.05)

## Evaluation

### Evaluate on PLEval-test

```bash
python scripts/evaluate.py \
    --config configs/finetune_config.yaml \
    --checkpoint checkpoints/finetune_best.pth \
    --output_dir results/evaluation \
    --save_vis \
    --num_vis 50
```

This computes:
- Overall mAP@50, mAP@50:95
- Seen categories mAP (21 classes)
- Unseen categories mAP (11 classes for zero-shot evaluation)
- Per-category AP breakdown

### Zero-shot Evaluation

```bash
python scripts/evaluate.py \
    --config configs/pretrain_config.yaml \
    --checkpoint checkpoints/pretrain_best.pth \
    --output_dir results/zeroshot \
    --mode zeroshot
```

## Inference

### Single Image

```bash
python scripts/inference.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/finetune_best.pth \
    --config configs/finetune_config.yaml \
    --categories configs/categories.yaml \
    --visualize \
    --save_json
```

### Batch Inference

```bash
python scripts/inference.py \
    --image_dir path/to/images \
    --checkpoint checkpoints/finetune_best.pth \
    --config configs/finetune_config.yaml \
    --categories configs/categories.yaml \
    --output_dir results/inference \
    --visualize \
    --benchmark
```

### Interactive Demo

```bash
# Single image with detailed visualization
python scripts/demo.py image \
    --image test.jpg \
    --checkpoint checkpoints/finetune_best.pth \
    --config configs/finetune_config.yaml \
    --categories configs/categories.yaml

# Confidence threshold comparison
python scripts/demo.py compare \
    --image test.jpg \
    --thresholds 0.1 0.25 0.5 0.75 \
    --checkpoint checkpoints/finetune_best.pth \
    --config configs/finetune_config.yaml \
    --categories configs/categories.yaml

# Video processing
python scripts/demo.py video \
    --video input.mp4 \
    --output output.mp4 \
    --checkpoint checkpoints/finetune_best.pth \
    --config configs/finetune_config.yaml \
    --categories configs/categories.yaml

# Zero-shot with custom categories
python scripts/demo.py zeroshot \
    --image test.jpg \
    --custom_categories "damaged insulator" "bird nest" "broken wire" \
    --checkpoint checkpoints/finetune_best.pth \
    --config configs/finetune_config.yaml \
    --categories configs/categories.yaml
```

## Ablation Studies

Run all ablation experiments as described in the paper:

```bash
# Run all ablation studies
python scripts/ablation/run_all_ablations.py \
    --config configs/ablation_config.yaml \
    --output_dir results/ablation \
    --epochs 30

# Run specific ablation
python scripts/ablation/run_all_ablations.py \
    --config configs/ablation_config.yaml \
    --single text_encoder

# Run subset of ablations
python scripts/ablation/run_all_ablations.py \
    --config configs/ablation_config.yaml \
    --studies pretraining text_encoder pavlpan
```

### Available Ablation Studies

| Study | Description | Key Finding |
|-------|-------------|-------------|
| `pretraining` | Effect of VL pretraining | +15.2% mAP with pretraining |
| `text_encoder` | BERT vs RoBERTa vs PowerBERT | PowerBERT +3.8% over baseline |
| `freeze_strategy` | Text encoder freezing | Freeze during pretrain, partial unfreeze during finetune |
| `pavlpan` | PA-VL-PAN components | PT-CSPLayer most critical |
| `backbone` | YOLOv12 components | Area Attention essential for elongated objects |
| `hierarchical_label` | Label hierarchy structure | Full hierarchy best for zero-shot |

## Model Zoo

| Model | Backbone | Params | GFLOPs | Zero-shot mAP | Fine-tuned mAP | Download |
|-------|----------|--------|--------|---------------|----------------|----------|
| PLVLDet-N | YOLOv12-N | 18.6M | 45.2 | 38.5% | 80.8% | [link]() |
| PLVLDet-S | YOLOv12-S | 32.4M | 78.5 | 41.2% | 82.3% | [link]() |
| PLVLDet-M | YOLOv12-M | 58.7M | 142.8 | 43.8% | 84.1% | [link]() |

## Supported Categories (32 Classes)

### Hierarchical Structure

```
├── Insulator (IT) - Parent
│   ├── Ceramic insulator (CI)
│   │   ├── Normal ceramic insulator (NCI)
│   │   ├── Contaminated ceramic insulator (CCI)
│   │   ├── Broken ceramic insulator (BCI)
│   │   └── Fall-off ceramic insulator (FoCI) [Unseen]
│   ├── Glass insulator (GI)
│   │   ├── Normal glass insulator (NGI)
│   │   ├── Broken glass insulator (BGI)
│   │   └── Contaminated glass insulator (CGI)
│   └── Polymer insulator (PI)
│       ├── Normal polymer insulator (NPI)
│       ├── Damaged polymer insulator (DaPI) [Unseen]
│       ├── Contaminated polymer insulator (CPI) [Unseen]
│       └── Deformed polymer insulator (DePI) [Unseen]
├── Foreign matters on tower (FMT) - Parent
│   ├── Bird's nest on tower (BNT)
│   ├── Plastic bag on tower (PBT)
│   ├── Kite on tower (KoT) [Unseen]
│   └── Tower balloon (TB) [Unseen]
├── Wire foreign matter (WFM) - Parent
│   ├── Wire plastic bag (WPB)
│   ├── Wire kite (WK) [Unseen]
│   └── Wire balloon (WB) [Unseen]
├── Vibration damper (VD) - Parent
│   ├── Displacement of damper (DD) [Unseen]
│   └── Missing damper (MD) [Unseen]
├── Transmission line
│   ├── Broken strand (BSTL)
│   └── Loose strand (LSTL)
└── Clamp
    ├── Tension clamp (TC)
    ├── Armour clamp (AC)
    └── Missing split pin (MSP) [Unseen]
```

**Seen categories (21)**: Used in MixPLOD pretraining
**Unseen categories (11)**: Zero-shot evaluation in PLEval

## Results

### Comparison with State-of-the-Art

| Method | Zero-shot mAP | Fine-tuned mAP | Unseen mAP |
|--------|---------------|----------------|------------|
| YOLO-World-M | 2.8% | 75.4% | 58.3% |
| Grounding DINO-T | 3.1% | 72.6% | 55.7% |
| Grounding DINO-B | 3.5% | 66.3% | 54.4% |
| **PLVLDet (Ours)** | **38.5%** | **80.8%** | **68.9%** |

## Citation

```bibtex
@article{plvldet2025,
  title={PLVLDet: A Power Line Vision-Language Detection Model with Hierarchical Cross-Modal Fusion for Transmission Line Component Recognition},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLOv12](https://github.com/ultralytics/ultralytics) for backbone architecture
- [Transformers](https://github.com/huggingface/transformers) for BERT implementation
- [mmyolo](https://github.com/open-mmlab/mmyolo) for detection framework reference

## Contact

For questions or issues, please open a GitHub issue or contact: xxx@xxx.edu.cn
