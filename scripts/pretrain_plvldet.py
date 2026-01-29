#!/usr/bin/env python3
"""
PLVLDet Vision-Language Pretraining

Pretrains the full PLVLDet model on MixPLOD dataset (seen categories only).
Learns vision-language alignment through contrastive learning.

Training specifications from paper:
- Dataset: MixPLOD (142 transmission lines, 21 categories)
- Epochs: 100
- Batch size: 8 per GPU (4 GPUs total = 32)
- Learning rate: 0.0002 with cosine decay
- Optimizer: AdamW
- Text encoder: Frozen
- Mosaic probability: 0.5
- Mixup probability: 0.1

Usage:
    # Single GPU
    python scripts/pretrain_plvldet.py \
        --config configs/pretrain_config.yaml
    
    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/pretrain_plvldet.py \
        --config configs/pretrain_config.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import math
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import build_plvldet
from data import VLDetectionDataset, build_train_transforms, detection_collate_fn
from losses import PLVLDetLoss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


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


class Trainer:
    """PLVLDet pretraining trainer."""
    
    def __init__(self, config, rank=0, world_size=1, local_rank=0):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = rank == 0
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')
            
        # Load category configuration
        self.category_config = load_category_config(config['data']['category_config'])
        
        # Build components
        self._build_model()
        self._build_datasets()
        self._build_optimizer()
        self._build_criterion()
        
        # Mixed precision
        self.scaler = GradScaler(enabled=config['training'].get('use_amp', True))
        
        # Training state
        self.start_epoch = 1
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Load checkpoint if specified
        if config.get('resume'):
            self._load_checkpoint(config['resume'])
            
    def _build_model(self):
        """Build PLVLDet model."""
        if self.is_main:
            logger.info("Building PLVLDet model...")
            
        model_config = self.config['model']
        
        self.model = build_plvldet(
            backbone=model_config.get('backbone', 'yolov12'),
            text_encoder=model_config.get('text_encoder', 'powerbert'),
            num_classes=len(self.category_config['categories']),
            embed_dim=model_config.get('embed_dim', 512),
            reg_max=model_config.get('reg_max', 16),
            depth_multiple=model_config.get('depth_multiple', 0.67),
            width_multiple=model_config.get('width_multiple', 0.75),
            pretrained_bert=model_config.get('pretrained_bert', None),
            pretrained_powerbert=self.config.get('powerbert_checkpoint', None)
        )
        
        # Freeze text encoder
        if self.config['training'].get('freeze_text_encoder', True):
            for param in self.model.text_encoder.parameters():
                param.requires_grad = False
            if self.is_main:
                logger.info("Text encoder frozen")
                
        self.model = self.model.to(self.device)
        
        # Distributed training
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
            
        # Count parameters
        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
            logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
            
    def _build_datasets(self):
        """Build training dataset and dataloader."""
        if self.is_main:
            logger.info("Building datasets...")
            
        data_config = self.config['data']
        training_config = self.config['training']
        
        # Build transforms
        train_transforms = build_train_transforms(
            img_size=data_config.get('img_size', 1000),
            mosaic_prob=training_config.get('mosaic_prob', 0.5),
            mixup_prob=training_config.get('mixup_prob', 0.1),
            hsv_h=training_config.get('hsv_h', 0.015),
            hsv_s=training_config.get('hsv_s', 0.7),
            hsv_v=training_config.get('hsv_v', 0.4),
            degrees=training_config.get('degrees', 0.0),
            translate=training_config.get('translate', 0.1),
            scale=training_config.get('scale', 0.5),
            shear=training_config.get('shear', 0.0),
            flipud=training_config.get('flipud', 0.0),
            fliplr=training_config.get('fliplr', 0.5)
        )
        
        # Build dataset
        self.train_dataset = VLDetectionDataset(
            image_dir=data_config['image_dir'],
            label_dir=data_config['label_dir'],
            split_file=data_config.get('split_file', None),
            category_config=self.category_config,
            transforms=train_transforms,
            img_size=data_config.get('img_size', 1000),
            include_text=True
        )
        
        if self.is_main:
            logger.info(f"Training samples: {len(self.train_dataset)}")
            
        # Build dataloader
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        else:
            sampler = None
            
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=data_config.get('num_workers', 8),
            pin_memory=True,
            collate_fn=detection_collate_fn,
            drop_last=True
        )
        
        self.sampler = sampler
        
    def _build_optimizer(self):
        """Build optimizer and scheduler."""
        training_config = self.config['training']
        
        # Separate parameters for different learning rates
        params = []
        
        # Get actual model (unwrap DDP if needed)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Backbone parameters (lower learning rate)
        backbone_params = list(model.backbone.parameters())
        params.append({
            'params': backbone_params,
            'lr': training_config['lr'] * 0.1
        })
        
        # Other parameters (full learning rate)
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'backbone' not in name:
                other_params.append(param)
        params.append({
            'params': other_params,
            'lr': training_config['lr']
        })
        
        self.optimizer = AdamW(
            params,
            lr=training_config['lr'],
            weight_decay=training_config.get('weight_decay', 0.0005),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        total_epochs = training_config['epochs']
        warmup_epochs = training_config.get('warmup_epochs', 3)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_epochs * len(self.train_loader)
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=(total_epochs - warmup_epochs) * len(self.train_loader),
            eta_min=training_config['lr'] * 0.01
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs * len(self.train_loader)]
        )
        
    def _build_criterion(self):
        """Build loss function."""
        loss_config = self.config['loss']
        
        self.criterion = PLVLDetLoss(
            contrastive_weight=loss_config.get('contrastive_weight', 1.0),
            box_weight=loss_config.get('box_weight', 7.5),
            dfl_weight=loss_config.get('dfl_weight', 1.5),
            reg_max=self.config['model'].get('reg_max', 16),
            temperature=loss_config.get('temperature', 0.07)
        )
        
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        if self.is_main:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.is_main:
            logger.info(f"Resumed from epoch {checkpoint['epoch']}")
            
    def _save_checkpoint(self, epoch, loss, is_best=False):
        """Save training checkpoint."""
        if not self.is_main:
            return
            
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        output_dir = Path(self.config['output']['checkpoint_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latest
        latest_path = output_dir / 'pretrain_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save periodic
        if epoch % self.config['output'].get('save_interval', 10) == 0:
            epoch_path = output_dir / f'pretrain_epoch{epoch}.pth'
            torch.save(checkpoint, epoch_path)
            
        # Save best
        if is_best:
            best_path = output_dir / 'pretrain_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss: {loss:.4f}")
            
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
            
        total_loss = 0
        total_loss_con = 0
        total_loss_box = 0
        total_loss_dfl = 0
        num_batches = 0
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Encode text embeddings for all categories
        with torch.no_grad():
            text_prompts = [
                cat['text_prompts'] for cat in self.category_config['categories']
            ]
            text_embed = model.encode_texts(text_prompts, device=self.device)
            
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}',
            disable=not self.is_main
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images'].to(self.device)
            targets = batch['targets']
            
            # Move targets to device
            for target in targets:
                for key in target:
                    if isinstance(target[key], torch.Tensor):
                        target[key] = target[key].to(self.device)
                        
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config['training'].get('use_amp', True)):
                outputs = self.model(images)
                outputs['text_embed'] = text_embed
                
                loss_dict = self.criterion(
                    outputs,
                    targets,
                    img_size=(images.shape[2], images.shape[3])
                )
                
                loss = loss_dict['loss']
                
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training'].get('grad_clip', 10.0)
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            total_loss_con += loss_dict['loss_con'].item()
            total_loss_box += loss_dict['loss_box'].item()
            total_loss_dfl += loss_dict['loss_dfl'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'con': f'{loss_dict["loss_con"].item():.4f}',
                'box': f'{loss_dict["loss_box"].item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
        # Average losses
        avg_loss = total_loss / num_batches
        avg_loss_con = total_loss_con / num_batches
        avg_loss_box = total_loss_box / num_batches
        avg_loss_dfl = total_loss_dfl / num_batches
        
        return {
            'loss': avg_loss,
            'loss_con': avg_loss_con,
            'loss_box': avg_loss_box,
            'loss_dfl': avg_loss_dfl
        }
        
    def train(self):
        """Run full training loop."""
        if self.is_main:
            logger.info("Starting pretraining...")
            logger.info(f"Total epochs: {self.config['training']['epochs']}")
            logger.info(f"Batch size per GPU: {self.config['training']['batch_size']}")
            logger.info(f"Total batch size: {self.config['training']['batch_size'] * self.world_size}")
            
        for epoch in range(self.start_epoch, self.config['training']['epochs'] + 1):
            # Train one epoch
            metrics = self.train_epoch(epoch)
            
            if self.is_main:
                logger.info(
                    f"Epoch {epoch} - "
                    f"Loss: {metrics['loss']:.4f} - "
                    f"Con: {metrics['loss_con']:.4f} - "
                    f"Box: {metrics['loss_box']:.4f} - "
                    f"DFL: {metrics['loss_dfl']:.4f}"
                )
                
                # Save checkpoint
                is_best = metrics['loss'] < self.best_loss
                if is_best:
                    self.best_loss = metrics['loss']
                    
                self._save_checkpoint(epoch, metrics['loss'], is_best)
                
        if self.is_main:
            logger.info("Pretraining complete!")
            logger.info(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='PLVLDet Vision-Language Pretraining')
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.resume:
        config['resume'] = args.resume
        
    try:
        # Create trainer and run
        trainer = Trainer(config, rank, world_size, local_rank)
        trainer.train()
        
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
