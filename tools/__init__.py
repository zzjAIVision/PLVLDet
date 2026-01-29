"""
PLVLDet Data Processing Tools

Command-line tools for dataset preparation:
- labelme_converter.py: Convert LabelMe annotations to COCO/YOLO format
- patch_generator.py: Generate 1000x1000 patches from large images
- dataset_splitter.py: Split dataset into MixPLOD and PLEval

Usage:
    # Step 1: Convert LabelMe annotations
    python tools/labelme_converter.py --image_dir data_images --label_dir data_labels
    
    # Step 2: Generate patches
    python tools/patch_generator.py --patch_size 1000 --stride 800
    
    # Step 3: Split into train/val/test
    python tools/dataset_splitter.py --pretrain_ratio 0.75
"""
