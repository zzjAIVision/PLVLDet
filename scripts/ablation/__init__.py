"""
Ablation Study Scripts for PLVLDet

This module provides scripts for conducting ablation experiments
as described in the paper. The ablation studies analyze contributions
of different components:

1. Vision-language pretraining effect (pretraining_ablation.py)
2. Text encoder comparison (text_encoder_ablation.py)
3. Text encoder freezing strategies (freeze_strategy_ablation.py)
4. PA-VL-PAN component analysis (pavlpan_ablation.py)
5. YOLOv12 backbone components (backbone_ablation.py)
6. Hierarchical label structure (hierarchical_label_ablation.py)

Usage:
    # Run all ablation studies
    python scripts/ablation/run_all_ablations.py --config configs/finetune_config.yaml
    
    # Run specific ablation study
    python scripts/ablation/run_all_ablations.py --config configs/finetune_config.yaml --single text_encoder
    
    # Run subset of studies
    python scripts/ablation/run_all_ablations.py --config configs/finetune_config.yaml \\
        --studies pretraining text_encoder pavlpan
"""

from .text_encoder_ablation import TextEncoderAblation
from .pavlpan_ablation import PAVLPANAblation
from .backbone_ablation import BackboneAblation
from .freeze_strategy_ablation import FreezeStrategyAblation
from .pretraining_ablation import PretrainingAblation
from .hierarchical_label_ablation import HierarchicalLabelAblation

__all__ = [
    'TextEncoderAblation',
    'PAVLPANAblation', 
    'BackboneAblation',
    'FreezeStrategyAblation',
    'PretrainingAblation',
    'HierarchicalLabelAblation',
]
