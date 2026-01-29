"""
PLVLDet: Power Line Vision-Language Detector

Full detection model combining:
- PowerBERT: Domain-adapted text encoder
- YOLOv12 Backbone: Visual feature extractor
- PA-VL-PAN: Vision-Language feature fusion network
- VL Detection Head: Vision-language detection head with region-text matching
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange

from .powerbert import PowerBERT, build_powerbert
from .backbone import YOLOv12Backbone, build_backbone
from .pa_vl_pan import PAVLPAN, build_neck


class DFL(nn.Module):
    """
    Distribution Focal Loss module for bounding box regression.
    
    Models bounding box coordinates as discrete probability distributions
    rather than deterministic values.
    """
    
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        # Initialize with discrete integral weights
        self.conv.weight.data = torch.arange(reg_max, dtype=torch.float).view(1, reg_max, 1, 1)
        self.conv.weight.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DFL.
        
        Args:
            x: Distribution predictions (B, reg_max*4, H, W)
            
        Returns:
            Box coordinates (B, 4, H, W)
        """
        B, _, H, W = x.shape
        x = x.view(B, 4, self.reg_max, H, W)
        x = F.softmax(x, dim=2)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (B, 4, H, W, reg_max)
        x = x.view(B * 4, H, W, self.reg_max).permute(0, 3, 1, 2)  # (B*4, reg_max, H, W)
        x = self.conv(x).view(B, 4, H, W)
        return x


class VLDetectionHead(nn.Module):
    """
    Vision-Language Detection Head.
    
    Generates:
    - Bounding box predictions via regression branch
    - Object embeddings via text contrastive branch
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 512,
        num_classes: int = 32,
        reg_max: int = 16,
        num_anchors: int = 1
    ):
        """
        Initialize VL Detection Head.
        
        Args:
            in_channels: Input channel count from PA-VL-PAN
            embed_dim: Object embedding dimension (same as text embedding)
            num_classes: Number of categories
            reg_max: Maximum value for DFL regression
            num_anchors: Number of anchors per position
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_anchors = num_anchors
        self.embed_dim = embed_dim
        
        # Shared stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        
        # Box regression branch
        self.box_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.box_head = nn.Conv2d(in_channels, 4 * reg_max, 1)
        
        # Object embedding branch (for region-text matching)
        self.embed_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.embed_head = nn.Conv2d(in_channels, embed_dim, 1)
        self.embed_norm = nn.LayerNorm(embed_dim)
        
        # Objectness branch
        self.obj_head = nn.Conv2d(in_channels, 1, 1)
        
        # DFL module for box decoding
        self.dfl = DFL(reg_max)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize head weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        # Initialize bias for objectness
        nn.init.constant_(self.obj_head.bias, -math.log((1 - 0.01) / 0.01))
        
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through detection head.
        
        Args:
            features: Multi-scale features from PA-VL-PAN [P3', P4', P5']
            
        Returns:
            Tuple of (box_preds, obj_preds, embed_preds, dfl_preds) for each scale
        """
        box_preds = []
        obj_preds = []
        embed_preds = []
        dfl_preds = []
        
        for feat in features:
            # Shared stem
            x = self.stem(feat)
            
            # Box regression
            box_feat = self.box_stem(x)
            box_raw = self.box_head(box_feat)  # (B, 4*reg_max, H, W)
            box_pred = self.dfl(box_raw)  # (B, 4, H, W)
            
            # Object embedding
            embed_feat = self.embed_stem(x)
            embed_raw = self.embed_head(embed_feat)  # (B, embed_dim, H, W)
            # Normalize embeddings
            B, D, H, W = embed_raw.shape
            embed_flat = rearrange(embed_raw, 'b d h w -> (b h w) d')
            embed_norm = self.embed_norm(embed_flat)
            embed_pred = rearrange(embed_norm, '(b h w) d -> b d h w', b=B, h=H, w=W)
            
            # Objectness
            obj_pred = self.obj_head(x)  # (B, 1, H, W)
            
            box_preds.append(box_pred)
            obj_preds.append(obj_pred)
            embed_preds.append(embed_pred)
            dfl_preds.append(box_raw)
            
        return box_preds, obj_preds, embed_preds, dfl_preds


class RegionTextMatcher(nn.Module):
    """
    Region-Text Matching module.
    
    Computes similarity between object region embeddings and 
    category text embeddings for classification.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        
    def forward(
        self,
        region_embed: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute region-text similarity scores.
        
        Args:
            region_embed: Region embeddings (B, D, H, W) or (N, D)
            text_embed: Text embeddings (C, D)
            
        Returns:
            Similarity logits (B, C, H, W) or (N, C)
        """
        spatial_input = region_embed.dim() == 4
        
        if spatial_input:
            B, D, H, W = region_embed.shape
            region_flat = rearrange(region_embed, 'b d h w -> (b h w) d')
        else:
            region_flat = region_embed
            
        # L2 normalize embeddings
        region_norm = F.normalize(region_flat, dim=-1)
        text_norm = F.normalize(text_embed, dim=-1)
        
        # Compute scaled dot product similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (region_norm @ text_norm.t())
        
        if spatial_input:
            C = text_embed.shape[0]
            logits = rearrange(logits, '(b h w) c -> b c h w', b=B, h=H, w=W)
            
        return logits


class PLVLDet(nn.Module):
    """
    PLVLDet: Power Line Vision-Language Detector.
    
    End-to-end vision-language detection model for transmission line
    component recognition, combining:
    - PowerBERT for domain-adapted text encoding
    - YOLOv12 backbone for visual feature extraction
    - PA-VL-PAN for hierarchical cross-modal fusion
    - VL Detection Head for region-text aligned detection
    """
    
    def __init__(
        self,
        text_encoder: PowerBERT,
        backbone: YOLOv12Backbone,
        neck: PAVLPAN,
        head: VLDetectionHead,
        category_texts: Dict[str, List[str]],
        num_classes: int = 32,
        strides: Tuple[int, ...] = (8, 16, 32)
    ):
        """
        Initialize PLVLDet.
        
        Args:
            text_encoder: PowerBERT text encoder
            backbone: YOLOv12 visual backbone
            neck: PA-VL-PAN feature fusion network
            head: VL Detection Head
            category_texts: Dictionary mapping category names to text prompts
            num_classes: Number of categories
            strides: Detection strides for each feature level
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        self.backbone = backbone
        self.neck = neck
        self.head = head
        
        self.category_texts = category_texts
        self.num_classes = num_classes
        self.strides = strides
        
        # Region-text matcher
        self.matcher = RegionTextMatcher()
        
        # Pre-computed text embeddings (cached for inference)
        self._text_embed_cache: Optional[torch.Tensor] = None
        
    def encode_texts(self, device: torch.device) -> torch.Tensor:
        """
        Encode category texts into embeddings.
        
        Args:
            device: Target device
            
        Returns:
            Text embedding matrix (num_classes, embed_dim)
        """
        if self._text_embed_cache is not None and not self.training:
            return self._text_embed_cache.to(device)
            
        text_embed = self.text_encoder.encode_categories(
            self.category_texts,
            device=device,
            use_cache=not self.training
        )
        
        if not self.training:
            self._text_embed_cache = text_embed.detach()
            
        return text_embed
        
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PLVLDet.
        
        Args:
            images: Input images (B, 3, H, W)
            targets: Optional list of target dictionaries for training
            
        Returns:
            Dictionary containing predictions or losses
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Encode category texts
        text_embed = self.encode_texts(device)  # (C, D)
        
        # Extract visual features
        backbone_feats = self.backbone(images)  # [C3, C4, C5]
        
        # Vision-language fusion
        neck_feats = self.neck(backbone_feats, text_embed)  # [P3', P4', P5']
        
        # Detection head
        box_preds, obj_preds, embed_preds, dfl_preds = self.head(neck_feats)
        
        # Compute region-text similarity for classification
        cls_preds = []
        for embed_pred in embed_preds:
            cls_logits = self.matcher(embed_pred, text_embed)
            cls_preds.append(cls_logits)
            
        outputs = {
            'box_preds': box_preds,      # List of (B, 4, H, W)
            'obj_preds': obj_preds,      # List of (B, 1, H, W)
            'cls_preds': cls_preds,      # List of (B, C, H, W)
            'embed_preds': embed_preds,  # List of (B, D, H, W)
            'dfl_preds': dfl_preds,      # List of (B, 4*reg_max, H, W)
            'text_embed': text_embed,    # (C, D)
            'strides': self.strides
        }
        
        return outputs
    
    def predict(
        self,
        images: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.65,
        max_det: int = 300
    ) -> List[Dict]:
        """
        Run inference and post-process predictions.
        
        Args:
            images: Input images (B, 3, H, W)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum detections per image
            
        Returns:
            List of detection results per image
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)
            
        return self._postprocess(
            outputs,
            images.shape[2:],
            conf_thres,
            iou_thres,
            max_det
        )
        
    def _postprocess(
        self,
        outputs: Dict,
        img_size: Tuple[int, int],
        conf_thres: float,
        iou_thres: float,
        max_det: int
    ) -> List[Dict]:
        """
        Post-process model outputs to get final detections.
        
        Args:
            outputs: Model outputs dictionary
            img_size: Input image size (H, W)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum detections
            
        Returns:
            List of detection results
        """
        from ..utils.postprocess import postprocess_detections
        
        return postprocess_detections(
            box_preds=outputs['box_preds'],
            obj_preds=outputs['obj_preds'],
            cls_preds=outputs['cls_preds'],
            strides=outputs['strides'],
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )


def build_plvldet(cfg: dict, category_texts: Dict[str, List[str]]) -> PLVLDet:
    """
    Build PLVLDet model from configuration.
    
    Args:
        cfg: Model configuration dictionary
        category_texts: Category text descriptions
        
    Returns:
        Initialized PLVLDet model
    """
    # Build text encoder
    text_encoder = build_powerbert(cfg['text_encoder'])
    
    # Build backbone
    backbone = build_backbone(cfg['backbone'])
    
    # Build neck
    neck_cfg = cfg['neck']
    neck_cfg['in_channels'] = backbone.out_channels
    neck = build_neck(neck_cfg)
    
    # Build head
    head_cfg = cfg['head']
    head = VLDetectionHead(
        in_channels=neck_cfg['out_channels'],
        embed_dim=head_cfg.get('embed_dim', 512),
        num_classes=head_cfg.get('num_classes', 32),
        reg_max=head_cfg.get('reg_max', 16)
    )
    
    # Build full model
    model = PLVLDet(
        text_encoder=text_encoder,
        backbone=backbone,
        neck=neck,
        head=head,
        category_texts=category_texts,
        num_classes=head_cfg.get('num_classes', 32)
    )
    
    return model
