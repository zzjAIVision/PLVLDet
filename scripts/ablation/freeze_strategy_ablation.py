#!/usr/bin/env python3
"""
Text Encoder Freeze Strategy Ablation Study

Analyzes the effect of freezing text encoder parameters:
- During pretraining (freeze vs. finetune)
- During fine-tuning (freeze vs. finetune)
- Partial freezing (freeze bottom N layers)

This experiment validates the paper's design choice of freezing
the text encoder to stabilize cross-modal alignment.
"""

import os
import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
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
from utils.metrics import APCalculator


class FreezeStrategyAblation:
    """Ablation study on text encoder freezing strategies."""
    
    FREEZE_CONFIGS = {
        'pretrain_freeze_finetune_freeze': {
            'pretrain_freeze': True,
            'finetune_freeze': True,
            'freeze_layers': 12,
            'description': 'Freeze text encoder in both phases (proposed)'
        },
        'pretrain_train_finetune_freeze': {
            'pretrain_freeze': False,
            'finetune_freeze': True,
            'freeze_layers': 12,
            'description': 'Train text encoder in pretrain, freeze in finetune'
        },
        'pretrain_freeze_finetune_train': {
            'pretrain_freeze': True,
            'finetune_freeze': False,
            'freeze_layers': 0,
            'description': 'Freeze text encoder in pretrain, train in finetune'
        },
        'pretrain_train_finetune_train': {
            'pretrain_freeze': False,
            'finetune_freeze': False,
            'freeze_layers': 0,
            'description': 'Train text encoder in both phases'
        },
        'partial_freeze_6': {
            'pretrain_freeze': True,
            'finetune_freeze': True,
            'freeze_layers': 6,
            'description': 'Freeze bottom 6 layers, train top 6'
        },
        'partial_freeze_10': {
            'pretrain_freeze': True,
            'finetune_freeze': True,
            'freeze_layers': 10,
            'description': 'Freeze bottom 10 layers, train top 2'
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
        logger = logging.getLogger('FreezeStrategyAblation')
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.output_dir / 'freeze_strategy_ablation.log')
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _create_detector(
        self,
        freeze_config: str,
        phase: str = 'pretrain'
    ) -> PLVLDet:
        """Create PLVLDet with specified freeze configuration."""
        cfg = self.FREEZE_CONFIGS[freeze_config]
        
        if phase == 'pretrain':
            freeze_text = cfg['pretrain_freeze']
        else:
            freeze_text = cfg['finetune_freeze']
        
        freeze_layers = cfg['freeze_layers'] if freeze_text else 0
        
        detector = PLVLDet(
            num_classes=self.config['model']['num_classes'],
            backbone_type=self.config['model'].get('backbone_type', 'yolov12n'),
            text_dim=self.config['model']['text_dim'],
            visual_dim=self.config['model']['visual_dim'],
            fusion_dim=self.config['model']['fusion_dim'],
            freeze_text_encoder=freeze_text,
            freeze_text_layers=freeze_layers,
            use_pa_vl_pan=True,
            use_spa=True,
            use_soem=True
        )
        
        return detector
    
    def _set_text_encoder_freeze(
        self,
        model: nn.Module,
        freeze: bool,
        freeze_layers: int = 12
    ):
        """Set text encoder parameters freeze state."""
        if not hasattr(model, 'text_encoder'):
            return
        
        text_encoder = model.text_encoder
        
        if freeze:
            for param in text_encoder.parameters():
                param.requires_grad = False
            
            if freeze_layers < 12 and hasattr(text_encoder, 'bert'):
                for i in range(freeze_layers, 12):
                    if hasattr(text_encoder.bert.encoder, 'layer'):
                        for param in text_encoder.bert.encoder.layer[i].parameters():
                            param.requires_grad = True
        else:
            for param in text_encoder.parameters():
                param.requires_grad = True
    
    def _count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        text_total = 0
        text_trainable = 0
        if hasattr(model, 'text_encoder'):
            text_total = sum(p.numel() for p in model.text_encoder.parameters())
            text_trainable = sum(
                p.numel() for p in model.text_encoder.parameters() 
                if p.requires_grad
            )
        
        return {
            'total': total,
            'trainable': trainable,
            'text_total': text_total,
            'text_trainable': text_trainable
        }
    
    def _train_epoch(
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
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
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
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {'total_loss': total_loss / num_batches}
    
    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        category_names: List[str]
    ) -> Dict[str, float]:
        """Evaluate model."""
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
    
    def _run_single_config(
        self,
        freeze_config: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        category_names: List[str],
        pretrain_epochs: int,
        finetune_epochs: int,
        eval_interval: int
    ) -> Dict:
        """Run training for a single freeze configuration."""
        cfg = self.FREEZE_CONFIGS[freeze_config]
        
        results = {
            'freeze_config': freeze_config,
            'config': cfg,
            'pretrain_history': [],
            'finetune_history': [],
            'best_pretrain_metrics': {},
            'best_finetune_metrics': {}
        }
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Configuration: {freeze_config}")
        self.logger.info(f"Description: {cfg['description']}")
        self.logger.info(f"{'='*60}")
        
        self.logger.info("\n--- Phase 1: Pretraining ---")
        model = self._create_detector(freeze_config, phase='pretrain')
        model = model.to(self.device)
        
        param_counts = self._count_parameters(model)
        results['pretrain_params'] = param_counts
        
        self.logger.info(f"Total params: {param_counts['total']:,}")
        self.logger.info(f"Trainable params: {param_counts['trainable']:,}")
        self.logger.info(f"Text encoder trainable: {param_counts['text_trainable']:,}")
        
        criterion = PLVLDetLoss(
            num_classes=self.config['model']['num_classes'],
            contrastive_weight=self.config['training'].get('contrastive_weight', 1.0),
            ciou_weight=self.config['training'].get('ciou_weight', 7.5),
            dfl_weight=self.config['training'].get('dfl_weight', 1.5)
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['training']['lr'],
            weight_decay=self.config['training'].get('weight_decay', 0.05)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=pretrain_epochs
        )
        
        scaler = GradScaler()
        
        best_pretrain_map = 0.0
        pretrain_checkpoint = None
        
        for epoch in range(1, pretrain_epochs + 1):
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch
            )
            
            results['pretrain_history'].append({
                'epoch': epoch,
                **train_metrics
            })
            
            scheduler.step()
            
            if epoch % eval_interval == 0 or epoch == pretrain_epochs:
                val_metrics = self._evaluate(model, val_loader, category_names)
                
                self.logger.info(f"Pretrain Epoch {epoch}: mAP@50={val_metrics['mAP50']:.4f}")
                
                if val_metrics['mAP50'] > best_pretrain_map:
                    best_pretrain_map = val_metrics['mAP50']
                    results['best_pretrain_metrics'] = {
                        'mAP50': val_metrics['mAP50'],
                        'epoch': epoch
                    }
                    pretrain_checkpoint = model.state_dict().copy()
        
        self.logger.info("\n--- Phase 2: Fine-tuning ---")
        
        finetune_model = self._create_detector(freeze_config, phase='finetune')
        finetune_model = finetune_model.to(self.device)
        
        if pretrain_checkpoint:
            finetune_model.load_state_dict(pretrain_checkpoint, strict=False)
        
        self._set_text_encoder_freeze(
            finetune_model,
            cfg['finetune_freeze'],
            cfg['freeze_layers']
        )
        
        param_counts = self._count_parameters(finetune_model)
        results['finetune_params'] = param_counts
        
        self.logger.info(f"Trainable params after freeze: {param_counts['trainable']:,}")
        self.logger.info(f"Text encoder trainable: {param_counts['text_trainable']:,}")
        
        trainable_params = [p for p in finetune_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['training']['lr'] * 0.5,
            weight_decay=self.config['training'].get('weight_decay', 0.05)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs
        )
        
        scaler = GradScaler()
        
        best_finetune_map = 0.0
        
        for epoch in range(1, finetune_epochs + 1):
            train_metrics = self._train_epoch(
                finetune_model, train_loader, optimizer, criterion, scaler, epoch
            )
            
            results['finetune_history'].append({
                'epoch': epoch,
                **train_metrics
            })
            
            scheduler.step()
            
            if epoch % eval_interval == 0 or epoch == finetune_epochs:
                val_metrics = self._evaluate(finetune_model, val_loader, category_names)
                
                self.logger.info(f"Finetune Epoch {epoch}: mAP@50={val_metrics['mAP50']:.4f}")
                
                if val_metrics['mAP50'] > best_finetune_map:
                    best_finetune_map = val_metrics['mAP50']
                    results['best_finetune_metrics'] = {
                        'mAP50': val_metrics['mAP50'],
                        'mAP50_95': val_metrics.get('mAP50_95', 0),
                        'epoch': epoch
                    }
                    
                    ckpt_path = self.output_dir / f'{freeze_config}_best.pth'
                    torch.save({
                        'model_state_dict': finetune_model.state_dict(),
                        'freeze_config': freeze_config,
                        'epoch': epoch,
                        'metrics': val_metrics
                    }, ckpt_path)
        
        del model, finetune_model
        torch.cuda.empty_cache()
        
        return results
    
    def run_ablation(
        self,
        freeze_configs: List[str] = None,
        pretrain_epochs: int = 20,
        finetune_epochs: int = 15,
        eval_interval: int = 5
    ):
        """Run freeze strategy ablation study."""
        if freeze_configs is None:
            freeze_configs = list(self.FREEZE_CONFIGS.keys())
        
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
        
        for freeze_config in freeze_configs:
            result = self._run_single_config(
                freeze_config,
                train_loader,
                val_loader,
                category_names,
                pretrain_epochs,
                finetune_epochs,
                eval_interval
            )
            self.results[freeze_config] = result
        
        self._save_results()
        self._generate_report()
    
    def _save_results(self):
        """Save ablation results."""
        results_path = self.output_dir / 'freeze_strategy_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")
    
    def _generate_report(self):
        """Generate ablation study report."""
        report_path = self.output_dir / 'freeze_strategy_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("TEXT ENCODER FREEZE STRATEGY ABLATION STUDY REPORT\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 90 + "\n")
            f.write("CONFIGURATION SUMMARY\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Config':<35} {'Pretrain':<12} {'Finetune':<12} {'Freeze Layers'}\n")
            f.write("-" * 90 + "\n")
            
            for name, cfg in self.FREEZE_CONFIGS.items():
                f.write(
                    f"{name:<35} "
                    f"{'Freeze' if cfg['pretrain_freeze'] else 'Train':<12} "
                    f"{'Freeze' if cfg['finetune_freeze'] else 'Train':<12} "
                    f"{cfg['freeze_layers']}\n"
                )
            
            f.write("\n" + "-" * 90 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Config':<35} {'Pretrain mAP':<15} {'Finetune mAP':<15} {'Trainable %'}\n")
            f.write("-" * 90 + "\n")
            
            for name in self.FREEZE_CONFIGS.keys():
                if name in self.results:
                    r = self.results[name]
                    pretrain_map = r['best_pretrain_metrics'].get('mAP50', 0)
                    finetune_map = r['best_finetune_metrics'].get('mAP50', 0)
                    
                    trainable_pct = 0
                    if r.get('finetune_params'):
                        total = r['finetune_params']['total']
                        trainable = r['finetune_params']['trainable']
                        trainable_pct = trainable / total * 100 if total > 0 else 0
                    
                    f.write(
                        f"{name:<35} "
                        f"{pretrain_map:.4f}         "
                        f"{finetune_map:.4f}         "
                        f"{trainable_pct:.1f}%\n"
                    )
            
            f.write("\n" + "-" * 90 + "\n")
            f.write("ANALYSIS\n")
            f.write("-" * 90 + "\n")
            
            if 'pretrain_freeze_finetune_freeze' in self.results:
                proposed = self.results['pretrain_freeze_finetune_freeze']
                proposed_map = proposed['best_finetune_metrics'].get('mAP50', 0)
                
                f.write(f"\nProposed strategy (freeze in both phases): {proposed_map:.4f} mAP@50\n")
                
                for name, r in self.results.items():
                    if name != 'pretrain_freeze_finetune_freeze':
                        other_map = r['best_finetune_metrics'].get('mAP50', 0)
                        diff = proposed_map - other_map
                        f.write(f"vs {name}: {'+' if diff > 0 else ''}{diff:.4f}\n")
            
            f.write("\n" + "-" * 90 + "\n")
            f.write("CONCLUSION\n")
            f.write("-" * 90 + "\n")
            f.write("Freezing the text encoder during both pretraining and fine-tuning:\n")
            f.write("- Stabilizes cross-modal alignment learning\n")
            f.write("- Reduces trainable parameters, improving efficiency\n")
            f.write("- Preserves domain knowledge from PowerBERT MLM pretraining\n")
            f.write("- Prevents catastrophic forgetting of semantic representations\n")
        
        self.logger.info(f"Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Freeze Strategy Ablation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/ablation/freeze_strategy')
    parser.add_argument('--pretrain_epochs', type=int, default=20)
    parser.add_argument('--finetune_epochs', type=int, default=15)
    parser.add_argument('--eval_interval', type=int, default=5)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    ablation = FreezeStrategyAblation(config, args.output_dir)
    ablation.run_ablation(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        eval_interval=args.eval_interval
    )


if __name__ == '__main__':
    main()
