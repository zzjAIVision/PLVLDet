#!/usr/bin/env python3
"""
Text Encoder Ablation Study

Compares detection performance using different text encoders:
- BERT-base-chinese (baseline)
- RoBERTa-base-chinese
- PowerBERT (domain-adapted, proposed)

This experiment validates that domain-specific pre-training improves
semantic understanding of power line terminology.
"""

import os
import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.detector import PLVLDet
from models.powerbert import PowerBERT
from data.dataset import VLDetectionDataset
from data.dataloader import create_dataloader
from data.transforms import get_train_transforms, get_val_transforms
from losses import PLVLDetLoss
from utils.metrics import compute_map, APCalculator


class TextEncoderAblation:
    """Ablation study comparing different text encoders."""
    
    ENCODER_CONFIGS = {
        'bert-base': {
            'pretrained_model': 'bert-base-chinese',
            'hidden_dim': 768,
            'num_layers': 12,
            'description': 'BERT-base-chinese without domain adaptation'
        },
        'roberta-base': {
            'pretrained_model': 'hfl/chinese-roberta-wwm-ext',
            'hidden_dim': 768,
            'num_layers': 12,
            'description': 'RoBERTa with whole word masking'
        },
        'powerbert': {
            'pretrained_model': 'bert-base-chinese',
            'powerbert_checkpoint': 'checkpoints/powerbert/powerbert_best.pth',
            'hidden_dim': 768,
            'num_layers': 12,
            'description': 'PowerBERT with domain-specific MLM pretraining'
        }
    }
    
    def __init__(self, config: Dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        
        self.results = {}
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('TextEncoderAblation')
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.output_dir / 'text_encoder_ablation.log')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _create_text_encoder(self, encoder_type: str) -> PowerBERT:
        """Create text encoder based on specified type."""
        encoder_cfg = self.ENCODER_CONFIGS[encoder_type]
        
        text_encoder = PowerBERT(
            pretrained_model=encoder_cfg['pretrained_model'],
            hidden_dim=encoder_cfg['hidden_dim'],
            output_dim=self.config['model']['text_dim'],
            freeze_layers=self.config['model'].get('freeze_text_layers', 10)
        )
        
        if encoder_type == 'powerbert':
            ckpt_path = encoder_cfg['powerbert_checkpoint']
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                text_encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.logger.info(f"Loaded PowerBERT checkpoint from {ckpt_path}")
            else:
                self.logger.warning(f"PowerBERT checkpoint not found: {ckpt_path}")
        
        return text_encoder
    
    def _create_detector(self, encoder_type: str) -> PLVLDet:
        """Create PLVLDet with specified text encoder."""
        text_encoder = self._create_text_encoder(encoder_type)
        
        detector = PLVLDet(
            num_classes=self.config['model']['num_classes'],
            backbone_type=self.config['model'].get('backbone_type', 'yolov12n'),
            text_encoder=text_encoder,
            text_dim=self.config['model']['text_dim'],
            visual_dim=self.config['model']['visual_dim'],
            fusion_dim=self.config['model']['fusion_dim'],
            num_heads=self.config['model'].get('num_heads', 8),
            use_pa_vl_pan=True,
            use_spa=True
        )
        
        return detector
    
    def _train_single_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: GradScaler,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        loss_components = {'contrastive': 0.0, 'ciou': 0.0, 'dfl': 0.0}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['targets']
            text_embeddings = batch['text_embeddings'].to(self.device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images, text_embeddings)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            for k in loss_components:
                if k in loss_dict:
                    loss_components[k] += loss_dict[k].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'con': f'{loss_dict.get("contrastive", 0):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {'total_loss': avg_loss, **avg_components}
    
    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        category_names: List[str]
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        
        ap_calculator = APCalculator(
            num_classes=len(category_names),
            iou_threshold=0.5
        )
        
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            text_embeddings = batch['text_embeddings'].to(self.device)
            targets = batch['targets']
            
            outputs = model(images, text_embeddings)
            predictions = model.postprocess(outputs, conf_thresh=0.25, iou_thresh=0.45)
            
            for pred, target in zip(predictions, targets):
                ap_calculator.update(pred, target)
        
        metrics = ap_calculator.compute()
        
        return metrics
    
    def run_ablation(
        self,
        encoder_types: List[str] = None,
        num_epochs: int = 30,
        eval_interval: int = 5
    ):
        """Run ablation study for specified encoder types."""
        if encoder_types is None:
            encoder_types = list(self.ENCODER_CONFIGS.keys())
        
        train_dataset = VLDetectionDataset(
            image_dir=self.config['data']['train_image_dir'],
            annotation_file=self.config['data']['train_annotation'],
            category_file=self.config['data']['category_file'],
            transforms=get_train_transforms(self.config['data']['img_size']),
            include_text=True
        )
        
        val_dataset = VLDetectionDataset(
            image_dir=self.config['data']['val_image_dir'],
            annotation_file=self.config['data']['val_annotation'],
            category_file=self.config['data']['category_file'],
            transforms=get_val_transforms(self.config['data']['img_size']),
            include_text=True
        )
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 4)
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 4)
        )
        
        category_names = train_dataset.category_names
        
        for encoder_type in encoder_types:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training with encoder: {encoder_type}")
            self.logger.info(f"Description: {self.ENCODER_CONFIGS[encoder_type]['description']}")
            self.logger.info(f"{'='*60}")
            
            model = self._create_detector(encoder_type)
            model = model.to(self.device)
            
            criterion = PLVLDetLoss(
                num_classes=self.config['model']['num_classes'],
                contrastive_weight=self.config['training'].get('contrastive_weight', 1.0),
                ciou_weight=self.config['training'].get('ciou_weight', 7.5),
                dfl_weight=self.config['training'].get('dfl_weight', 1.5)
            )
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['training']['lr'],
                weight_decay=self.config['training'].get('weight_decay', 0.05)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=self.config['training']['lr'] * 0.01
            )
            
            scaler = GradScaler()
            
            encoder_results = {
                'encoder_type': encoder_type,
                'config': self.ENCODER_CONFIGS[encoder_type],
                'training_history': [],
                'validation_history': [],
                'best_map50': 0.0,
                'best_epoch': 0
            }
            
            best_map = 0.0
            
            for epoch in range(1, num_epochs + 1):
                train_metrics = self._train_single_epoch(
                    model, train_loader, optimizer, criterion, scaler, epoch
                )
                
                encoder_results['training_history'].append({
                    'epoch': epoch,
                    **train_metrics
                })
                
                scheduler.step()
                
                if epoch % eval_interval == 0 or epoch == num_epochs:
                    val_metrics = self._evaluate(model, val_loader, category_names)
                    
                    encoder_results['validation_history'].append({
                        'epoch': epoch,
                        **val_metrics
                    })
                    
                    self.logger.info(
                        f"Epoch {epoch}: mAP@50={val_metrics['mAP50']:.4f}, "
                        f"mAP@50:95={val_metrics.get('mAP50_95', 0):.4f}"
                    )
                    
                    if val_metrics['mAP50'] > best_map:
                        best_map = val_metrics['mAP50']
                        encoder_results['best_map50'] = best_map
                        encoder_results['best_epoch'] = epoch
                        
                        ckpt_path = self.output_dir / f'{encoder_type}_best.pth'
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'encoder_type': encoder_type,
                            'epoch': epoch,
                            'mAP50': best_map
                        }, ckpt_path)
            
            self.results[encoder_type] = encoder_results
            
            self.logger.info(f"\nEncoder {encoder_type} completed:")
            self.logger.info(f"  Best mAP@50: {encoder_results['best_map50']:.4f}")
            self.logger.info(f"  Best epoch: {encoder_results['best_epoch']}")
            
            del model
            torch.cuda.empty_cache()
        
        self._save_results()
        self._generate_report()
    
    def _save_results(self):
        """Save ablation results to JSON."""
        results_path = self.output_dir / 'text_encoder_ablation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")
    
    def _generate_report(self):
        """Generate human-readable ablation report."""
        report_path = self.output_dir / 'text_encoder_ablation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TEXT ENCODER ABLATION STUDY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Training Epochs: {self.config['training'].get('epochs', 30)}\n")
            f.write(f"Batch Size: {self.config['training']['batch_size']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Encoder':<20} {'mAP@50':<12} {'Best Epoch':<12} {'Description'}\n")
            f.write("-" * 70 + "\n")
            
            sorted_results = sorted(
                self.results.items(),
                key=lambda x: x[1]['best_map50'],
                reverse=True
            )
            
            for encoder_type, result in sorted_results:
                desc = self.ENCODER_CONFIGS[encoder_type]['description'][:30]
                f.write(
                    f"{encoder_type:<20} "
                    f"{result['best_map50']:.4f}      "
                    f"{result['best_epoch']:<12} "
                    f"{desc}\n"
                )
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("ANALYSIS\n")
            f.write("-" * 70 + "\n")
            
            if len(sorted_results) >= 2:
                best_encoder = sorted_results[0][0]
                baseline = 'bert-base' if 'bert-base' in self.results else sorted_results[-1][0]
                
                if baseline in self.results:
                    improvement = (
                        self.results[best_encoder]['best_map50'] - 
                        self.results[baseline]['best_map50']
                    )
                    f.write(f"\nBest performing encoder: {best_encoder}\n")
                    f.write(f"Improvement over {baseline}: +{improvement:.4f} mAP@50\n")
                    
                    if best_encoder == 'powerbert':
                        f.write("\nConclusion: Domain-adapted PowerBERT achieves superior \n")
                        f.write("performance, validating that MLM pretraining on power system \n")
                        f.write("terminology improves semantic understanding of equipment categories.\n")
        
        self.logger.info(f"Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Text Encoder Ablation Study')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='results/ablation/text_encoder',
                        help='Output directory for results')
    parser.add_argument('--encoders', type=str, nargs='+',
                        default=['bert-base', 'roberta-base', 'powerbert'],
                        help='Text encoders to compare')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluation interval in epochs')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    ablation = TextEncoderAblation(config, args.output_dir)
    ablation.run_ablation(
        encoder_types=args.encoders,
        num_epochs=args.epochs,
        eval_interval=args.eval_interval
    )


if __name__ == '__main__':
    main()
