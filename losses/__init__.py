"""
Loss Functions for PLVLDet

This module implements the training losses:
- Region-Text Contrastive Loss: Aligns region embeddings with text embeddings
- Complete IoU (CIoU) Loss: Bounding box regression
- Distribution Focal Loss (DFL): Probabilistic box coordinate prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange


class RegionTextContrastiveLoss(nn.Module):
    """
    Region-Text Contrastive Loss.
    
    Aligns visual region embeddings with category text embeddings through
    contrastive learning. Positive pairs are matched based on ground truth
    categories, negative samples are drawn from other categories.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize Region-Text Contrastive Loss.
        
        Args:
            temperature: Temperature coefficient for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        region_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            region_embeds: Region embeddings (N, D)
            text_embeds: Text embeddings (C, D)
            labels: Ground truth class indices (N,)
            
        Returns:
            Scalar loss value
        """
        # L2 normalize embeddings
        region_embeds = F.normalize(region_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Compute similarity logits
        logits = region_embeds @ text_embeds.t() / self.temperature  # (N, C)
        
        # Cross-entropy loss with ground truth labels
        loss = F.cross_entropy(logits, labels)
        
        return loss


class CIoULoss(nn.Module):
    """
    Complete IoU (CIoU) Loss for bounding box regression.
    
    Accounts for overlap ratio, center distance, and aspect ratio consistency.
    Particularly effective for power line components with diverse aspect ratios.
    """
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CIoU loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format
            target_boxes: Target boxes (N, 4) in xyxy format
            
        Returns:
            CIoU loss value
        """
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=-1)
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_boxes.unbind(dim=-1)
        
        # Compute areas
        pred_w = (pred_x2 - pred_x1).clamp(min=0)
        pred_h = (pred_y2 - pred_y1).clamp(min=0)
        tgt_w = (tgt_x2 - tgt_x1).clamp(min=0)
        tgt_h = (tgt_y2 - tgt_y1).clamp(min=0)
        
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h
        
        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Union and IoU
        union_area = pred_area + tgt_area - inter_area + self.eps
        iou = inter_area / union_area
        
        # Smallest enclosing box
        enclose_x1 = torch.min(pred_x1, tgt_x1)
        enclose_y1 = torch.min(pred_y1, tgt_y1)
        enclose_x2 = torch.max(pred_x2, tgt_x2)
        enclose_y2 = torch.max(pred_y2, tgt_y2)
        
        enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
        
        # Diagonal length of enclosing box
        c2 = enclose_w ** 2 + enclose_h ** 2 + self.eps
        
        # Center distance
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        tgt_cx = (tgt_x1 + tgt_x2) / 2
        tgt_cy = (tgt_y1 + tgt_y2) / 2
        
        rho2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2
        
        # Aspect ratio penalty
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(tgt_w / (tgt_h + self.eps)) - 
            torch.atan(pred_w / (pred_h + self.eps)), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
            
        # CIoU
        ciou = iou - (rho2 / c2) - alpha * v
        
        return 1 - ciou.mean()


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) for probabilistic bounding box regression.
    
    Models box coordinates as discrete probability distributions rather than
    deterministic values, allowing the network to express uncertainty.
    """
    
    def __init__(self, reg_max: int = 16):
        """
        Initialize DFL.
        
        Args:
            reg_max: Maximum value for discretization bins
        """
        super().__init__()
        self.reg_max = reg_max
        
    def forward(
        self,
        pred_dist: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DFL loss.
        
        Args:
            pred_dist: Predicted distributions (N, reg_max)
            target: Target values (N,) - continuous values to be discretized
            
        Returns:
            DFL loss value
        """
        # Convert continuous targets to distribution labels
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        
        # Left and right bin indices
        left = target.long()
        right = left + 1
        
        # Weights for left and right bins
        weight_left = right.float() - target
        weight_right = target - left.float()
        
        # Cross-entropy with soft labels
        loss = (
            F.cross_entropy(pred_dist, left, reduction='none') * weight_left +
            F.cross_entropy(pred_dist, right, reduction='none') * weight_right
        )
        
        return loss.mean()


class PLVLDetLoss(nn.Module):
    """
    Combined loss for PLVLDet training.
    
    Integrates:
    - Region-Text Contrastive Loss for vision-language alignment
    - CIoU Loss for bounding box regression
    - DFL for probabilistic coordinate prediction
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        reg_max: int = 16,
        temperature: float = 0.07
    ):
        """
        Initialize combined loss.
        
        Args:
            contrastive_weight: Weight for contrastive loss
            box_weight: Weight for CIoU loss
            dfl_weight: Weight for DFL loss
            reg_max: Maximum value for DFL discretization
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        
        self.contrastive_loss = RegionTextContrastiveLoss(temperature)
        self.ciou_loss = CIoULoss()
        self.dfl_loss = DistributionFocalLoss(reg_max)
        
        self.reg_max = reg_max
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict],
        img_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: List of target dictionaries per image
            img_size: Input image size (H, W)
            
        Returns:
            Dictionary of loss components and total loss
        """
        device = outputs['text_embed'].device
        batch_size = len(targets)
        
        # Initialize accumulators
        loss_con = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        
        num_pos = 0
        
        # Get predictions at each scale
        box_preds = outputs['box_preds']
        cls_preds = outputs['cls_preds']
        embed_preds = outputs['embed_preds']
        dfl_preds = outputs['dfl_preds']
        text_embed = outputs['text_embed']
        strides = outputs['strides']
        
        # Process each scale
        for scale_idx, stride in enumerate(strides):
            box_pred = box_preds[scale_idx]
            cls_pred = cls_preds[scale_idx]
            embed_pred = embed_preds[scale_idx]
            dfl_pred = dfl_preds[scale_idx]
            
            B, _, H, W = box_pred.shape
            
            # Assign targets to grid positions
            for batch_idx in range(B):
                if batch_idx >= len(targets) or len(targets[batch_idx].get('boxes', [])) == 0:
                    continue
                    
                target = targets[batch_idx]
                gt_boxes = target['boxes'].to(device)  # (M, 4)
                gt_labels = target['labels'].to(device)  # (M,)
                
                # Convert boxes to grid coordinates
                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 / stride
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 / stride
                
                # Get grid positions
                gt_gi = gt_cx.long().clamp(0, W - 1)
                gt_gj = gt_cy.long().clamp(0, H - 1)
                
                for obj_idx in range(len(gt_boxes)):
                    gi, gj = gt_gi[obj_idx], gt_gj[obj_idx]
                    
                    # Get predictions at this position
                    pred_box = box_pred[batch_idx, :, gj, gi]  # (4,)
                    pred_embed = embed_pred[batch_idx, :, gj, gi]  # (D,)
                    pred_dfl = dfl_pred[batch_idx, :, gj, gi]  # (4*reg_max,)
                    
                    # Ground truth
                    gt_box = gt_boxes[obj_idx]  # (4,)
                    gt_label = gt_labels[obj_idx]  # scalar
                    
                    # Convert predictions to absolute coordinates
                    # Anchor-free: predict offsets from grid cell
                    pred_x1 = (gi - pred_box[0]) * stride
                    pred_y1 = (gj - pred_box[1]) * stride
                    pred_x2 = (gi + pred_box[2]) * stride
                    pred_y2 = (gj + pred_box[3]) * stride
                    pred_box_abs = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2])
                    
                    # CIoU loss
                    loss_box = loss_box + self.ciou_loss(
                        pred_box_abs.unsqueeze(0),
                        gt_box.unsqueeze(0)
                    )
                    
                    # DFL loss
                    # Convert ground truth offsets to DFL targets
                    gt_offsets = torch.stack([
                        gi - gt_box[0] / stride,
                        gj - gt_box[1] / stride,
                        gt_box[2] / stride - gi,
                        gt_box[3] / stride - gj
                    ])
                    
                    pred_dfl_reshaped = pred_dfl.view(4, self.reg_max)
                    for i in range(4):
                        loss_dfl = loss_dfl + self.dfl_loss(
                            pred_dfl_reshaped[i:i+1],
                            gt_offsets[i:i+1]
                        )
                    
                    # Contrastive loss
                    loss_con = loss_con + self.contrastive_loss(
                        pred_embed.unsqueeze(0),
                        text_embed,
                        gt_label.unsqueeze(0)
                    )
                    
                    num_pos += 1
                    
        # Average over positive samples
        num_pos = max(num_pos, 1)
        loss_con = loss_con / num_pos
        loss_box = loss_box / num_pos
        loss_dfl = loss_dfl / num_pos
        
        # Total weighted loss
        total_loss = (
            self.contrastive_weight * loss_con +
            self.box_weight * loss_box +
            self.dfl_weight * loss_dfl
        )
        
        return {
            'loss': total_loss,
            'loss_con': loss_con,
            'loss_box': loss_box,
            'loss_dfl': loss_dfl,
            'num_pos': num_pos
        }


def build_loss(cfg: dict) -> PLVLDetLoss:
    """
    Build PLVLDet loss from configuration.
    
    Args:
        cfg: Loss configuration dictionary
        
    Returns:
        Initialized PLVLDetLoss
    """
    con_cfg = cfg.get('contrastive', {})
    box_cfg = cfg.get('box', {})
    dfl_cfg = cfg.get('dfl', {})
    
    return PLVLDetLoss(
        contrastive_weight=con_cfg.get('weight', 1.0),
        box_weight=box_cfg.get('weight', 7.5),
        dfl_weight=dfl_cfg.get('weight', 1.5),
        reg_max=dfl_cfg.get('reg_max', 16),
        temperature=con_cfg.get('temperature', 0.07)
    )
