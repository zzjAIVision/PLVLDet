"""
PLVLDet Training, Evaluation, and Demo Scripts

This package contains all scripts for training, evaluation, ablation studies,
inference, and demonstration of the PLVLDet model.

Scripts Overview:
-----------------

Training Pipeline:
    pretrain_powerbert.py   - Stage 1: PowerBERT domain adaptation via MLM
    pretrain_plvldet.py     - Stage 2: Vision-language pretraining on MixPLOD
    finetune_plvldet.py     - Stage 3: Fine-tuning on PLEval dataset

Evaluation:
    evaluate.py             - Evaluate model performance on test set

Inference:
    inference.py            - Run inference on single images or directories

Demonstration:
    demo.py                 - Interactive demo with various visualization modes

Ablation Studies:
    ablation/               - Directory containing all ablation study scripts
        run_all_ablations.py        - Run all or selected ablation studies
        text_encoder_ablation.py    - Compare text encoders
        pavlpan_ablation.py         - Analyze PA-VL-PAN components
        backbone_ablation.py        - Compare backbone architectures
        freeze_strategy_ablation.py - Text encoder freezing strategies
        pretraining_ablation.py     - Effect of vision-language pretraining
        hierarchical_label_ablation.py - Hierarchical label structure analysis

Quick Start:
-----------

# 1. Data Preparation
python tools/labelme_converter.py --image_dir data_images --label_dir data_labels \\
    --output_dir data/annotations --format coco
python tools/patch_generator.py --image_dir data_images \\
    --annotation_file data/annotations/annotations.json --output_dir data/patches
python tools/dataset_splitter.py --patch_dir data/patches --output_dir data/splits

# 2. Training
# Stage 1: PowerBERT
python scripts/pretrain_powerbert.py --corpus_path data/power_corpus.txt \\
    --output_dir checkpoints/powerbert

# Stage 2: Vision-language pretraining
torchrun --nproc_per_node=4 scripts/pretrain_plvldet.py \\
    --config configs/pretrain_config.yaml

# Stage 3: Fine-tuning
torchrun --nproc_per_node=4 scripts/finetune_plvldet.py \\
    --config configs/finetune_config.yaml \\
    --pretrained checkpoints/pretrain_best.pth

# 3. Evaluation
python scripts/evaluate.py --config configs/finetune_config.yaml \\
    --checkpoint checkpoints/finetune_best.pth --save_vis

# 4. Inference
python scripts/inference.py --image test_image.jpg \\
    --checkpoint checkpoints/finetune_best.pth \\
    --config configs/finetune_config.yaml \\
    --categories configs/categories.yaml --visualize

# 5. Demo
python scripts/demo.py image --image test_image.jpg \\
    --checkpoint checkpoints/finetune_best.pth \\
    --config configs/finetune_config.yaml \\
    --categories configs/categories.yaml

# 6. Ablation Studies
python scripts/ablation/run_all_ablations.py \\
    --config configs/ablation_config.yaml --epochs 30
"""
