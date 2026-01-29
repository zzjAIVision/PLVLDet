#!/usr/bin/env python3
"""
PowerBERT Domain Adaptation Pretraining

Continues pretraining BERT-base-Chinese on power domain corpus using
Masked Language Modeling (MLM) to adapt the text encoder to power line
inspection terminology.

Training specifications from paper:
- Base model: BERT-base-Chinese
- MLM probability: 0.15
- Epochs: 30
- Batch size: 32
- Learning rate: 2e-5
- Weight decay: 0.01

Usage:
    python scripts/pretrain_powerbert.py \
        --corpus_path data/power_corpus.txt \
        --output_dir checkpoints/powerbert \
        --epochs 30 \
        --batch_size 32
"""

import os
import sys
import argparse
import logging
import math
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.powerbert import PowerBERTForMLM


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Power domain vocabulary for generating synthetic corpus
POWER_DOMAIN_TERMS = {
    'components': [
        '绝缘子', '瓷绝缘子', '玻璃绝缘子', '复合绝缘子', '正常绝缘子',
        '破损绝缘子', '污秽绝缘子', '变形绝缘子', '脱落绝缘子',
        '防振锤', '位移防振锤', '缺失防振锤', '线夹', '耐张线夹', '铠装线夹',
        '导线', '断股', '散股', '输电线', '杆塔', '横担', '地线',
        'insulator', 'ceramic insulator', 'glass insulator', 'polymer insulator',
        'vibration damper', 'tension clamp', 'armour clamp', 'conductor',
        'transmission line', 'tower', 'cross arm', 'ground wire'
    ],
    'defects': [
        '破损', '污秽', '变形', '脱落', '位移', '缺失', '异物',
        '鸟巢', '塑料袋', '风筝', '气球', '开口销',
        'broken', 'contaminated', 'deformed', 'fall-off', 'displacement',
        'missing', 'foreign matter', "bird's nest", 'plastic bag', 'kite',
        'balloon', 'split pin'
    ],
    'inspection': [
        '巡检', '检测', '识别', '定位', '缺陷', '故障', '异常',
        '输电线路', '电力系统', '高压', '超高压', '特高压',
        'inspection', 'detection', 'recognition', 'localization', 'defect',
        'fault', 'abnormal', 'transmission line', 'power system',
        'high voltage', 'extra high voltage', 'ultra high voltage'
    ]
}


class PowerCorpusDataset(Dataset):
    """Dataset for power domain MLM pretraining."""
    
    def __init__(self, corpus_path, tokenizer, max_length=128, mlm_prob=0.15):
        """
        Initialize corpus dataset.
        
        Args:
            corpus_path: Path to corpus file (one sentence per line)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            mlm_prob: Probability of masking tokens
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_prob = mlm_prob
        
        # Load corpus
        self.sentences = []
        corpus_path = Path(corpus_path)
        
        if corpus_path.exists():
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) > 10:  # Filter short lines
                        self.sentences.append(line)
        else:
            # Generate synthetic corpus from domain vocabulary
            logger.warning(f"Corpus file not found: {corpus_path}")
            logger.info("Generating synthetic power domain corpus...")
            self.sentences = self._generate_synthetic_corpus(5000)
            
        logger.info(f"Loaded {len(self.sentences)} sentences for MLM pretraining")
        
    def _generate_synthetic_corpus(self, num_samples):
        """Generate synthetic sentences from domain vocabulary."""
        sentences = []
        
        templates_zh = [
            "发现{component}存在{defect}问题，需要进行{inspection}处理。",
            "在输电线路巡检中，检测到{component}出现{defect}情况。",
            "{component}的{defect}缺陷可能导致线路故障。",
            "通过无人机巡检发现杆塔上有{defect}，影响{component}正常运行。",
            "对输电线路{component}进行定期{inspection}，及时发现{defect}隐患。",
            "电力系统中{component}的{inspection}工作十分重要。",
            "高压输电线路{component}容易出现{defect}问题。",
        ]
        
        templates_en = [
            "The {component} shows signs of {defect}, requiring immediate {inspection}.",
            "During transmission line patrol, {defect} was detected on the {component}.",
            "{defect} on the {component} may cause line failure.",
            "Drone inspection revealed {defect} affecting the {component} operation.",
            "Regular {inspection} of {component} helps identify {defect} issues early.",
            "{inspection} of power line {component} is critical for system reliability.",
            "High voltage transmission {component} is prone to {defect} problems.",
        ]
        
        for _ in range(num_samples):
            if random.random() < 0.6:  # 60% Chinese
                template = random.choice(templates_zh)
                component = random.choice(POWER_DOMAIN_TERMS['components'][:17])
                defect = random.choice(POWER_DOMAIN_TERMS['defects'][:12])
                inspection = random.choice(POWER_DOMAIN_TERMS['inspection'][:7])
            else:  # 40% English
                template = random.choice(templates_en)
                component = random.choice(POWER_DOMAIN_TERMS['components'][17:])
                defect = random.choice(POWER_DOMAIN_TERMS['defects'][12:])
                inspection = random.choice(POWER_DOMAIN_TERMS['inspection'][7:])
                
            sentence = template.format(
                component=component,
                defect=defect,
                inspection=inspection
            )
            sentences.append(sentence)
            
        return sentences
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        text = self.sentences[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create MLM labels
        labels = input_ids.clone()
        
        # Random mask positions (excluding special tokens)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        
        # Don't mask special tokens
        special_tokens_mask = torch.tensor([
            1 if token_id in [self.tokenizer.cls_token_id, 
                              self.tokenizer.sep_token_id,
                              self.tokenizer.pad_token_id] else 0
            for token_id in input_ids.tolist()
        ], dtype=torch.bool)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% MASK, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # Calculate accuracy on masked tokens
            predictions = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (predictions == labels) & mask
            
            total_correct += correct.sum().item()
            total_masked += mask.sum().item()
            num_batches += 1
            
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }


def main():
    parser = argparse.ArgumentParser(description='PowerBERT MLM Pretraining')
    
    # Data arguments
    parser.add_argument('--corpus_path', type=str, default='data/power_corpus.txt',
                        help='Path to power domain corpus')
    parser.add_argument('--output_dir', type=str, default='checkpoints/powerbert',
                        help='Output directory for checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--mlm_prob', type=float, default=0.15,
                        help='MLM masking probability')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    
    # Model arguments
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese',
                        help='Base BERT model name')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Output embedding dimension')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    logger.info(f"Initializing PowerBERT from {args.bert_model}...")
    model = PowerBERTForMLM(
        bert_model_name=args.bert_model,
        embed_dim=args.embed_dim
    )
    model = model.to(device)
    
    # Get tokenizer from model
    tokenizer = model.tokenizer
    
    # Create dataset
    logger.info("Loading corpus...")
    full_dataset = PowerCorpusDataset(
        corpus_path=args.corpus_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mlm_prob=args.mlm_prob
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.01
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f} - "
            f"Val PPL: {val_metrics['perplexity']:.2f}"
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = output_dir / 'powerbert_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, best_path)
            logger.info(f"Saved best model to {best_path}")
            
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = output_dir / f'powerbert_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss']
            }, ckpt_path)
            
    # Save final model
    final_path = output_dir / 'powerbert_final.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'bert_model': args.bert_model,
            'embed_dim': args.embed_dim,
            'max_length': args.max_length
        }
    }, final_path)
    
    logger.info(f"Training complete. Final model saved to {final_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
