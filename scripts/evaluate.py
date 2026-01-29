#!/usr/bin/env python3
"""
PLVLDet Evaluation Script

Evaluates trained PLVLDet model on test set, computing mAP and
per-category metrics for both seen and unseen categories.

Evaluation specifications from paper:
- Test set: PLEval-test (47 transmission lines)
- Metrics: mAP@50, mAP@50:95
- Zero-shot evaluation: Test on unseen categories
- Fine-tuned evaluation: Test on all 32 categories

Usage:
    python scripts/evaluate.py \
        --config configs/finetune_config.yaml \
        --checkpoint checkpoints/finetune_best.pth \
        --output_dir results/evaluation
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import build_plvldet
from data import VLDetectionDataset, build_val_transforms, detection_collate_fn
from utils.metrics import DetectionEvaluator
from utils.postprocess import postprocess_predictions
from utils.visualization import visualize_predictions, save_detection_results


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Category splits
SEEN_CATEGORIES = {0, 1, 2, 3, 4, 5, 11, 12, 15, 16, 17, 20, 21, 24, 25, 26, 27, 29, 30}
UNSEEN_CATEGORIES = {6, 7, 8, 9, 10, 13, 14, 18, 19, 22, 23, 28, 31}

CATEGORY_NAMES = [
    'Insulator', 'Ceramic insulator', 'Glass insulator', 'Polymer insulator',
    'Normal glass insulator', 'Broken glass insulator', 'Contaminated glass insulator',
    'Normal Polymer insulator', 'Damaged Polymer insulator', 'Contaminated Polymer insulator',
    'Deformed Polymer insulator', 'Normal ceramic insulator', 'Contaminated ceramic insulator',
    'Broken ceramic insulator', 'Fall-off ceramic insulator', 'Foreign matters on tower',
    "Bird's nest on tower", 'Plastic bag on tower', 'Kite on tower', 'Tower balloon',
    'Wire foreign matter', 'Wire plastic bag', 'Wire kite', 'Wire balloon',
    'Broken strand of transmission line', 'Loose strand of transmission line',
    'vibration damper', 'Displacement of damper', 'Missing damper',
    'Tension clamp', 'armour clamp', 'Missing split pin'
]


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_category_config(config_path):
    """Load category configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Evaluator:
    """PLVLDet evaluation class."""
    
    def __init__(self, config, checkpoint_path, output_dir):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load category configuration
        self.category_config = load_category_config(config['data']['category_config'])
        self.num_classes = len(self.category_config['categories'])
        
        # Build model and load weights
        self._build_model()
        
        # Build test dataset
        self._build_dataset()
        
        # Setup evaluators
        self.evaluator_all = DetectionEvaluator(
            num_classes=self.num_classes,
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
        self.evaluator_seen = DetectionEvaluator(
            num_classes=self.num_classes,
            iou_thresholds=[0.5]
        )
        self.evaluator_unseen = DetectionEvaluator(
            num_classes=self.num_classes,
            iou_thresholds=[0.5]
        )
        
    def _build_model(self):
        """Build model and load checkpoint."""
        logger.info("Building PLVLDet model...")
        
        model_config = self.config['model']
        
        self.model = build_plvldet(
            backbone=model_config.get('backbone', 'yolov12'),
            text_encoder=model_config.get('text_encoder', 'powerbert'),
            num_classes=self.num_classes,
            embed_dim=model_config.get('embed_dim', 512),
            reg_max=model_config.get('reg_max', 16),
            depth_multiple=model_config.get('depth_multiple', 0.67),
            width_multiple=model_config.get('width_multiple', 0.75)
        )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Handle DDP wrapped state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
    def _build_dataset(self):
        """Build test dataset."""
        logger.info("Building test dataset...")
        
        data_config = self.config['data']
        
        val_transforms = build_val_transforms(
            img_size=data_config.get('img_size', 1000)
        )
        
        self.test_dataset = VLDetectionDataset(
            image_dir=data_config['image_dir'],
            label_dir=data_config['label_dir'],
            split_file=data_config.get('test_split', None),
            category_config=self.category_config,
            transforms=val_transforms,
            img_size=data_config.get('img_size', 1000),
            include_text=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['evaluation'].get('batch_size', 8),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=detection_collate_fn
        )
        
        logger.info(f"Test samples: {len(self.test_dataset)}")
        
    @torch.no_grad()
    def evaluate(self, save_visualizations=False, num_vis_samples=50):
        """Run evaluation on test set."""
        logger.info("Starting evaluation...")
        
        # Encode text embeddings
        text_prompts = [
            cat['text_prompts'] for cat in self.category_config['categories']
        ]
        text_embed = self.model.encode_texts(text_prompts, device=self.device)
        
        # Reset evaluators
        self.evaluator_all.reset()
        self.evaluator_seen.reset()
        self.evaluator_unseen.reset()
        
        all_predictions = []
        vis_count = 0
        
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Evaluating')):
            images = batch['images'].to(self.device)
            targets = batch['targets']
            image_ids = batch.get('image_ids', list(range(len(targets))))
            
            # Forward pass
            outputs = self.model(images)
            outputs['text_embed'] = text_embed
            
            # Post-process predictions
            predictions = postprocess_predictions(
                outputs,
                text_embed,
                conf_thresh=self.config['evaluation'].get('conf_thresh', 0.001),
                iou_thresh=self.config['evaluation'].get('iou_thresh', 0.65),
                max_det=self.config['evaluation'].get('max_det', 300)
            )
            
            # Process each sample
            for idx, (pred, target) in enumerate(zip(predictions, targets)):
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                if len(pred) > 0:
                    pred_boxes = pred[:, :4].cpu().numpy()
                    pred_scores = pred[:, 4].cpu().numpy()
                    pred_labels = pred[:, 5].cpu().numpy().astype(int)
                else:
                    pred_boxes = np.array([])
                    pred_scores = np.array([])
                    pred_labels = np.array([])
                    
                # Add to all evaluator
                self.evaluator_all.add_sample(
                    pred_boxes, pred_scores, pred_labels,
                    gt_boxes, gt_labels
                )
                
                # Add to seen evaluator (filter by seen categories)
                seen_gt_mask = np.isin(gt_labels, list(SEEN_CATEGORIES))
                seen_pred_mask = np.isin(pred_labels, list(SEEN_CATEGORIES)) if len(pred_labels) > 0 else np.array([])
                
                self.evaluator_seen.add_sample(
                    pred_boxes[seen_pred_mask] if len(seen_pred_mask) > 0 else np.array([]),
                    pred_scores[seen_pred_mask] if len(seen_pred_mask) > 0 else np.array([]),
                    pred_labels[seen_pred_mask] if len(seen_pred_mask) > 0 else np.array([]),
                    gt_boxes[seen_gt_mask],
                    gt_labels[seen_gt_mask]
                )
                
                # Add to unseen evaluator (filter by unseen categories)
                unseen_gt_mask = np.isin(gt_labels, list(UNSEEN_CATEGORIES))
                unseen_pred_mask = np.isin(pred_labels, list(UNSEEN_CATEGORIES)) if len(pred_labels) > 0 else np.array([])
                
                self.evaluator_unseen.add_sample(
                    pred_boxes[unseen_pred_mask] if len(unseen_pred_mask) > 0 else np.array([]),
                    pred_scores[unseen_pred_mask] if len(unseen_pred_mask) > 0 else np.array([]),
                    pred_labels[unseen_pred_mask] if len(unseen_pred_mask) > 0 else np.array([]),
                    gt_boxes[unseen_gt_mask],
                    gt_labels[unseen_gt_mask]
                )
                
                # Store predictions
                all_predictions.append({
                    'image_id': image_ids[idx] if idx < len(image_ids) else batch_idx * len(targets) + idx,
                    'predictions': {
                        'boxes': pred_boxes.tolist(),
                        'scores': pred_scores.tolist(),
                        'labels': pred_labels.tolist()
                    },
                    'ground_truth': {
                        'boxes': gt_boxes.tolist(),
                        'labels': gt_labels.tolist()
                    }
                })
                
                # Save visualizations
                if save_visualizations and vis_count < num_vis_samples:
                    vis_dir = self.output_dir / 'visualizations'
                    vis_dir.mkdir(exist_ok=True)
                    
                    img = images[idx].cpu()
                    vis_path = vis_dir / f'sample_{vis_count:04d}.jpg'
                    
                    visualize_predictions(
                        img,
                        pred_boxes, pred_scores, pred_labels,
                        gt_boxes, gt_labels,
                        CATEGORY_NAMES,
                        vis_path
                    )
                    vis_count += 1
                    
        # Compute metrics
        metrics_all = self.evaluator_all.compute_metrics()
        metrics_seen = self.evaluator_seen.compute_metrics()
        metrics_unseen = self.evaluator_unseen.compute_metrics()
        
        # Compile results
        results = {
            'all_categories': metrics_all,
            'seen_categories': metrics_seen,
            'unseen_categories': metrics_unseen,
            'config': self.config,
            'checkpoint': self.checkpoint_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print results
        self._print_results(results)
        
        # Save results
        self._save_results(results, all_predictions)
        
        return results
        
    def _print_results(self, results):
        """Print evaluation results."""
        print("\n" + "=" * 70)
        print("PLVLDet Evaluation Results")
        print("=" * 70)
        
        print("\n--- All Categories (32 classes) ---")
        print(f"  mAP@50:      {results['all_categories'].get('mAP@50', 0):.4f}")
        print(f"  mAP@50:95:   {results['all_categories'].get('mAP@50:95', 0):.4f}")
        
        print("\n--- Seen Categories (21 classes) ---")
        print(f"  mAP@50:      {results['seen_categories'].get('mAP@50', 0):.4f}")
        
        print("\n--- Unseen Categories (Zero-shot, 14 classes) ---")
        print(f"  mAP@50:      {results['unseen_categories'].get('mAP@50', 0):.4f}")
        
        # Per-category AP
        print("\n--- Per-Category AP@50 ---")
        ap_per_class = results['all_categories'].get('ap_per_class', {})
        
        print("\nSeen categories:")
        for cat_id in sorted(SEEN_CATEGORIES):
            if cat_id in ap_per_class:
                ap = ap_per_class[cat_id]
                print(f"  [{cat_id:2d}] {CATEGORY_NAMES[cat_id]:40s}: {ap:.4f}")
                
        print("\nUnseen categories:")
        for cat_id in sorted(UNSEEN_CATEGORIES):
            if cat_id in ap_per_class:
                ap = ap_per_class[cat_id]
                print(f"  [{cat_id:2d}] {CATEGORY_NAMES[cat_id]:40s}: {ap:.4f}")
                
        print("\n" + "=" * 70)
        
    def _save_results(self, results, predictions):
        """Save evaluation results to files."""
        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy values to float for JSON serialization
            json_results = self._convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Save predictions
        predictions_path = self.output_dir / 'predictions.json'
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to: {predictions_path}")
        
        # Save summary report
        report_path = self.output_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write("PLVLDet Evaluation Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            f.write("Overall Results\n")
            f.write("-" * 40 + "\n")
            f.write(f"All Categories mAP@50:     {results['all_categories'].get('mAP@50', 0):.4f}\n")
            f.write(f"All Categories mAP@50:95:  {results['all_categories'].get('mAP@50:95', 0):.4f}\n")
            f.write(f"Seen Categories mAP@50:    {results['seen_categories'].get('mAP@50', 0):.4f}\n")
            f.write(f"Unseen Categories mAP@50:  {results['unseen_categories'].get('mAP@50', 0):.4f}\n\n")
            
            f.write("Per-Category Results\n")
            f.write("-" * 40 + "\n")
            ap_per_class = results['all_categories'].get('ap_per_class', {})
            
            f.write("\nSeen categories:\n")
            for cat_id in sorted(SEEN_CATEGORIES):
                if cat_id in ap_per_class:
                    ap = ap_per_class[cat_id]
                    f.write(f"  [{cat_id:2d}] {CATEGORY_NAMES[cat_id]:40s}: {ap:.4f}\n")
                    
            f.write("\nUnseen categories (zero-shot):\n")
            for cat_id in sorted(UNSEEN_CATEGORIES):
                if cat_id in ap_per_class:
                    ap = ap_per_class[cat_id]
                    f.write(f"  [{cat_id:2d}] {CATEGORY_NAMES[cat_id]:40s}: {ap:.4f}\n")
                    
        logger.info(f"Report saved to: {report_path}")
        
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description='PLVLDet Evaluation')
    parser.add_argument('--config', type=str, default='configs/finetune_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--num_vis', type=int, default=50,
                        help='Number of visualization samples')
    parser.add_argument('--conf_thresh', type=float, default=None,
                        help='Override confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=None,
                        help='Override IoU threshold')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override evaluation parameters if specified
    if 'evaluation' not in config:
        config['evaluation'] = {}
    if args.conf_thresh is not None:
        config['evaluation']['conf_thresh'] = args.conf_thresh
    if args.iou_thresh is not None:
        config['evaluation']['iou_thresh'] = args.iou_thresh
        
    # Create evaluator and run
    evaluator = Evaluator(config, args.checkpoint, args.output_dir)
    results = evaluator.evaluate(
        save_visualizations=args.save_vis,
        num_vis_samples=args.num_vis
    )


if __name__ == '__main__':
    main()
