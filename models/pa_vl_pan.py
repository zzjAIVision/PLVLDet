"""
PA-VL-PAN: Power-Aware Vision-Language Path Aggregation Network

This module implements the hierarchical cross-modal fusion network that combines
visual features with text embeddings for vision-language object detection.

Key components:
- PT-CSPLayer: Power-aware Text-guided CSP Layer with cross-attention
- Strip Pooling Attention (SPA): Long-range spatial modeling for elongated objects
- Small Object Enhancement Module (SOEM): Enhanced detection for small components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange


class ConvModule(nn.Module):
    """Standard convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: bool = True
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvModule(in_channels, hidden, 1)
        self.cv2 = ConvModule(hidden, out_channels, 3, groups=groups)
        self.add = shortcut and in_channels == out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class StripPoolingAttention(nn.Module):
    """
    Strip Pooling Attention (SPA) module.
    
    Captures long-range dependencies along horizontal and vertical directions
    through strip pooling operations, particularly effective for detecting
    elongated transmission line components.
    
    Complexity: O(H*W) compared to O((H*W)^2) for standard self-attention.
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        """
        Initialize Strip Pooling Attention.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for intermediate features
        """
        super().__init__()
        
        hidden_channels = max(in_channels // reduction, 32)
        
        # Horizontal strip pooling branch
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Vertical strip pooling branch
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Strip Pooling Attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted output of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Horizontal strip pooling: aggregate along height
        # Pool to (B, C, 1, W) then expand
        pool_h = F.adaptive_avg_pool2d(x, (1, W))  # (B, C, 1, W)
        attn_h = self.conv_h(pool_h)  # (B, C, 1, W)
        attn_h = attn_h.expand(-1, -1, H, -1)  # (B, C, H, W)
        
        # Vertical strip pooling: aggregate along width
        # Pool to (B, C, H, 1) then expand
        pool_v = F.adaptive_avg_pool2d(x, (H, 1))  # (B, C, H, 1)
        attn_v = self.conv_v(pool_v)  # (B, C, H, 1)
        attn_v = attn_v.expand(-1, -1, -1, W)  # (B, C, H, W)
        
        # Combine horizontal and vertical attention
        attn = self.sigmoid(attn_h + attn_v)
        
        # Apply attention
        return x * attn


class TextGuidedCrossAttention(nn.Module):
    """
    Multi-head Text-guided Cross-Attention mechanism.
    
    Visual features serve as queries, text embeddings as keys and values.
    This enables category-aware feature enhancement.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """
        Initialize Text-guided Cross-Attention.
        
        Args:
            visual_dim: Visual feature dimension
            text_dim: Text embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in projections
            attn_drop: Attention dropout rate
            proj_drop: Output dropout rate
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from visual features
        self.q_proj = nn.Linear(visual_dim, visual_dim, bias=qkv_bias)
        
        # Key and Value from text embeddings
        self.k_proj = nn.Linear(text_dim, visual_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(text_dim, visual_dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(visual_dim, visual_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(
        self,
        visual_feat: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention.
        
        Args:
            visual_feat: Visual features (B, H*W, D) or (B, D, H, W)
            text_embed: Text embeddings (num_classes, text_dim)
            
        Returns:
            Text-enhanced visual features of same shape as input
        """
        # Handle spatial input
        spatial_input = visual_feat.dim() == 4
        if spatial_input:
            B, D, H, W = visual_feat.shape
            visual_feat = rearrange(visual_feat, 'b d h w -> b (h w) d')
        else:
            B, N, D = visual_feat.shape
            
        # Expand text embeddings to batch
        num_classes = text_embed.shape[0]
        text_embed = text_embed.unsqueeze(0).expand(B, -1, -1)  # (B, C, text_dim)
        
        # Compute Q, K, V
        q = self.q_proj(visual_feat)  # (B, N, D)
        k = self.k_proj(text_embed)   # (B, C, D)
        v = self.v_proj(text_embed)   # (B, C, D)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b c (h d) -> b h c d', h=self.num_heads)
        v = rearrange(v, 'b c (h d) -> b h c d', h=self.num_heads)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, C)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        out = attn @ v  # (B, h, N, d)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Reshape back to spatial if needed
        if spatial_input:
            out = rearrange(out, 'b (h w) d -> b d h w', h=H, w=W)
            
        return out


class MaxSigmoidAttention(nn.Module):
    """
    Max-Sigmoid Attention mechanism.
    
    Computes category-level similarity scores between visual and text
    representations, using max pooling across categories followed by sigmoid.
    """
    
    def __init__(self, visual_dim: int, text_dim: int):
        super().__init__()
        
        # Align dimensions if needed
        if visual_dim != text_dim:
            self.align = nn.Linear(visual_dim, text_dim)
        else:
            self.align = nn.Identity()
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(
        self,
        visual_feat: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Max-Sigmoid Attention.
        
        Args:
            visual_feat: Visual features (B, D, H, W)
            text_embed: Text embeddings (num_classes, text_dim)
            
        Returns:
            Attention-weighted visual features (B, D, H, W)
        """
        B, D, H, W = visual_feat.shape
        
        # Align visual features
        visual_flat = rearrange(visual_feat, 'b d h w -> b (h w) d')
        visual_aligned = self.align(visual_flat)  # (B, H*W, text_dim)
        
        # Compute similarity with all category texts
        # text_embed: (C, text_dim)
        similarity = torch.einsum('bnd,cd->bnc', visual_aligned, text_embed)  # (B, H*W, C)
        
        # Max across categories
        max_sim, _ = similarity.max(dim=-1, keepdim=True)  # (B, H*W, 1)
        
        # Sigmoid activation
        attn = self.sigmoid(max_sim)  # (B, H*W, 1)
        
        # Reshape and apply
        attn = rearrange(attn, 'b (h w) 1 -> b 1 h w', h=H, w=W)
        
        return visual_feat * attn


class PTCSPLayer(nn.Module):
    """
    Power-aware Text-guided CSP Layer (PT-CSPLayer).
    
    Core component for vision-language feature fusion in PA-VL-PAN.
    Combines text-guided cross-attention with spatial feature extraction
    through bottleneck modules, fused via Max-Sigmoid attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        text_dim: int = 512,
        num_bottlenecks: int = 3,
        num_heads: int = 8,
        expansion: float = 0.5
    ):
        """
        Initialize PT-CSPLayer.
        
        Args:
            in_channels: Input channel count
            out_channels: Output channel count
            text_dim: Text embedding dimension
            num_bottlenecks: Number of bottleneck blocks
            num_heads: Number of attention heads
            expansion: Channel expansion ratio
        """
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # Channel reduction
        self.cv1 = ConvModule(in_channels, hidden_channels * 2, 1)
        
        # Upper branch: Text-guided Cross-Attention
        self.cross_attn = TextGuidedCrossAttention(
            visual_dim=hidden_channels,
            text_dim=text_dim,
            num_heads=num_heads
        )
        
        # Lower branch: Bottleneck modules
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut=True)
            for _ in range(num_bottlenecks)
        ])
        
        # Max-Sigmoid Attention
        self.max_sigmoid = MaxSigmoidAttention(hidden_channels, text_dim)
        
        # Channel merge
        self.cv2 = ConvModule(hidden_channels * 3, out_channels, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through PT-CSPLayer.
        
        Args:
            x: Visual features (B, C, H, W)
            text_embed: Text embeddings (num_classes, text_dim)
            
        Returns:
            Enhanced visual features (B, out_channels, H, W)
        """
        # Channel reduction and split
        x = self.cv1(x)
        x1, x2 = x.chunk(2, dim=1)
        
        # Cross-attention branch
        y1 = self.cross_attn(x1, text_embed)
        
        # Bottleneck branch
        y2 = self.bottlenecks(x2)
        
        # Max-Sigmoid attention
        y3 = self.max_sigmoid(x2, text_embed)
        
        # Merge all branches
        out = torch.cat([y1, y2, y3], dim=1)
        out = self.cv2(out)
        
        return out


class SmallObjectEnhancementModule(nn.Module):
    """
    Small Object Enhancement Module (SOEM).
    
    Enhances detection of small-scale components like vibration dampers,
    split pins, and fittings through multi-scale feature refinement.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize SOEM.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
        """
        super().__init__()
        
        # Multi-scale convolutions with different receptive fields
        self.conv3 = ConvModule(in_channels, out_channels // 4, 3, padding=1)
        self.conv5 = ConvModule(in_channels, out_channels // 4, 5, padding=2)
        self.conv7 = ConvModule(in_channels, out_channels // 4, 7, padding=3)
        
        # Dilated convolution for larger receptive field
        self.dilated = ConvModule(in_channels, out_channels // 4, 3, padding=2)
        self.dilated.conv.dilation = (2, 2)
        self.dilated.conv.padding = (2, 2)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            ConvModule(out_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        # Spatial attention for small object localization
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SOEM.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Enhanced features (B, out_channels, H, W)
        """
        # Multi-scale feature extraction
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        fd = self.dilated(x)
        
        # Concatenate multi-scale features
        out = torch.cat([f3, f5, f7, fd], dim=1)
        out = self.fusion(out)
        
        # Spatial attention
        avg_pool = out.mean(dim=1, keepdim=True)
        max_pool = out.max(dim=1, keepdim=True)[0]
        spatial = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attn = self.spatial_attn(spatial)
        
        return out * spatial_attn


class PAVLPAN(nn.Module):
    """
    Power-Aware Vision-Language Path Aggregation Network (PA-VL-PAN).
    
    Bidirectional feature pyramid with vision-language fusion:
    - Top-down path: Propagates rich semantic information
    - Bottom-up path: Enhances fine-grained spatial details
    - PT-CSPLayer: Text-guided cross-attention at each level
    - SPA: Strip Pooling Attention for elongated objects
    - SOEM: Small Object Enhancement for small components
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 512,
        text_dim: int = 512,
        num_csp_blocks: int = 3,
        use_strip_pooling: bool = True,
        use_soem: bool = True
    ):
        """
        Initialize PA-VL-PAN.
        
        Args:
            in_channels: Input channel counts [C3, C4, C5]
            out_channels: Output channel count for all levels
            text_dim: Text embedding dimension
            num_csp_blocks: Number of bottleneck blocks in PT-CSPLayer
            use_strip_pooling: Whether to use Strip Pooling Attention
            use_soem: Whether to use Small Object Enhancement Module
        """
        super().__init__()
        
        self.use_strip_pooling = use_strip_pooling
        self.use_soem = use_soem
        
        # Channel alignment for backbone features
        self.reduce_c5 = ConvModule(in_channels[2], out_channels, 1)
        self.reduce_c4 = ConvModule(in_channels[1], out_channels, 1)
        self.reduce_c3 = ConvModule(in_channels[0], out_channels, 1)
        
        # Top-down path
        self.pt_csp_p5 = PTCSPLayer(out_channels, out_channels, text_dim, num_csp_blocks)
        self.pt_csp_p4 = PTCSPLayer(out_channels * 2, out_channels, text_dim, num_csp_blocks)
        self.pt_csp_p3 = PTCSPLayer(out_channels * 2, out_channels, text_dim, num_csp_blocks)
        
        # Strip Pooling Attention modules
        if use_strip_pooling:
            self.spa_p5 = StripPoolingAttention(out_channels)
            self.spa_p4 = StripPoolingAttention(out_channels)
            self.spa_p3 = StripPoolingAttention(out_channels)
        
        # Small Object Enhancement Module for P3
        if use_soem:
            self.soem = SmallObjectEnhancementModule(out_channels, out_channels)
            
        # Bottom-up path
        self.down_p3_p4 = ConvModule(out_channels, out_channels, 3, stride=2)
        self.pt_csp_p4_up = PTCSPLayer(out_channels * 2, out_channels, text_dim, num_csp_blocks)
        
        self.down_p4_p5 = ConvModule(out_channels, out_channels, 3, stride=2)
        self.pt_csp_p5_up = PTCSPLayer(out_channels * 2, out_channels, text_dim, num_csp_blocks)
        
        # Upsample layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(
        self,
        features: List[torch.Tensor],
        text_embed: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Forward pass through PA-VL-PAN.
        
        Args:
            features: Multi-scale backbone features [C3, C4, C5]
            text_embed: Text embeddings (num_classes, text_dim)
            
        Returns:
            Enhanced multi-scale features [P3', P4', P5']
        """
        c3, c4, c5 = features
        
        # Channel alignment
        c5 = self.reduce_c5(c5)
        c4 = self.reduce_c4(c4)
        c3 = self.reduce_c3(c3)
        
        # ============ Top-down path ============
        # P5
        p5 = self.pt_csp_p5(c5, text_embed)
        if self.use_strip_pooling:
            p5 = self.spa_p5(p5)
            
        # P4
        p5_up = self.upsample(p5)
        p4 = torch.cat([c4, p5_up], dim=1)
        p4 = self.pt_csp_p4(p4, text_embed)
        if self.use_strip_pooling:
            p4 = self.spa_p4(p4)
            
        # P3
        p4_up = self.upsample(p4)
        p3 = torch.cat([c3, p4_up], dim=1)
        p3 = self.pt_csp_p3(p3, text_embed)
        if self.use_strip_pooling:
            p3 = self.spa_p3(p3)
            
        # ============ Bottom-up path ============
        # P3' with SOEM
        if self.use_soem:
            p3_prime = self.soem(p3)
        else:
            p3_prime = p3
            
        # P4'
        p3_down = self.down_p3_p4(p3_prime)
        p4_prime = torch.cat([p4, p3_down], dim=1)
        p4_prime = self.pt_csp_p4_up(p4_prime, text_embed)
        
        # P5'
        p4_down = self.down_p4_p5(p4_prime)
        p5_prime = torch.cat([p5, p4_down], dim=1)
        p5_prime = self.pt_csp_p5_up(p5_prime, text_embed)
        
        return [p3_prime, p4_prime, p5_prime]


def build_neck(cfg: dict) -> PAVLPAN:
    """
    Build PA-VL-PAN from configuration.
    
    Args:
        cfg: Neck configuration dictionary
        
    Returns:
        Initialized PAVLPAN module
    """
    return PAVLPAN(
        in_channels=cfg.get('in_channels', [256, 512, 1024]),
        out_channels=cfg.get('out_channels', 512),
        text_dim=cfg.get('text_dim', 512),
        num_csp_blocks=cfg.get('num_csp_blocks', 3),
        use_strip_pooling=cfg.get('strip_pooling', True),
        use_soem=cfg.get('soem_enabled', True)
    )
