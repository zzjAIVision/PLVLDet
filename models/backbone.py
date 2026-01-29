"""
YOLOv12 Backbone with R-ELAN and Area Attention

This module implements the visual feature extractor based on YOLOv12 architecture,
featuring Residual Efficient Layer Aggregation Network (R-ELAN) and Area Attention
for efficient long-range dependency modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from einops import rearrange


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """Calculate 'same' padding for convolution."""
    if padding is None:
        padding = dilation * (kernel_size - 1) // 2
    return padding


class ConvModule(nn.Module):
    """
    Standard convolution block with Conv2d + BatchNorm + Activation.
    
    This is the basic building block for the backbone network.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        act: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise convolution for implicit positional encoding."""
    
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, 1, padding, groups=dim, bias=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class AreaAttention(nn.Module):
    """
    Area Attention mechanism for efficient long-range dependency modeling.
    
    Splits the feature map into non-overlapping regions and performs 
    self-attention within each region, reducing computational complexity
    from O((H*W)^2) to O((H*W)^2 / L).
    
    This is particularly effective for detecting elongated transmission line
    components such as wires, insulator strings, and tower structures.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_regions: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = True
    ):
        """
        Initialize Area Attention.
        
        Args:
            dim: Input feature dimension
            num_heads: Number of attention heads
            num_regions: Number of regions to split (L in paper)
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            use_flash_attn: Whether to use Flash Attention
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_regions = num_regions
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Implicit positional encoding via depthwise convolution
        self.pe = DWConv(dim, kernel_size=7)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Area Attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Add positional encoding
        x = x + self.pe(x)
        
        # Reshape to (B, H*W, C)
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Compute QKV
        qkv = self.qkv(x_flat)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', 
                       three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Split into regions for area attention
        # We use horizontal splitting by default (good for horizontal structures)
        region_size = (H * W) // self.num_regions
        
        # Perform attention within each region
        if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's efficient attention implementation
            outputs = []
            for i in range(self.num_regions):
                start_idx = i * region_size
                end_idx = start_idx + region_size if i < self.num_regions - 1 else H * W
                
                q_region = q[:, :, start_idx:end_idx, :]
                k_region = k[:, :, start_idx:end_idx, :]
                v_region = v[:, :, start_idx:end_idx, :]
                
                # Efficient attention
                out_region = F.scaled_dot_product_attention(
                    q_region, k_region, v_region,
                    dropout_p=self.attn_drop.p if self.training else 0.0
                )
                outputs.append(out_region)
            
            out = torch.cat(outputs, dim=2)
        else:
            # Standard attention implementation
            outputs = []
            for i in range(self.num_regions):
                start_idx = i * region_size
                end_idx = start_idx + region_size if i < self.num_regions - 1 else H * W
                
                q_region = q[:, :, start_idx:end_idx, :]
                k_region = k[:, :, start_idx:end_idx, :]
                v_region = v[:, :, start_idx:end_idx, :]
                
                # Compute attention scores
                attn = (q_region @ k_region.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                
                out_region = attn @ v_region
                outputs.append(out_region)
            
            out = torch.cat(outputs, dim=2)
        
        # Reshape and project
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Reshape back to spatial
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvModule(in_channels, hidden_channels, 1)
        self.cv2 = ConvModule(hidden_channels, out_channels, 3, groups=groups)
        self.add = shortcut and in_channels == out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RELAN(nn.Module):
    """
    Residual Efficient Layer Aggregation Network (R-ELAN).
    
    Implements split-transform-merge structure with residual connections
    at block level, incorporating Area Attention for long-range context.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        use_area_attn: bool = True,
        area_attn_cfg: Optional[dict] = None
    ):
        """
        Initialize R-ELAN block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of bottleneck blocks in transform branch
            use_area_attn: Whether to use Area Attention
            area_attn_cfg: Configuration for Area Attention
        """
        super().__init__()
        
        hidden_channels = out_channels // 2
        
        # Split branches
        self.cv1 = ConvModule(in_channels, hidden_channels, 1)
        self.cv2 = ConvModule(in_channels, hidden_channels, 1)
        
        # Transform branch with Area Attention
        self.use_area_attn = use_area_attn
        if use_area_attn:
            attn_cfg = area_attn_cfg or {}
            self.area_attn = AreaAttention(
                dim=hidden_channels,
                num_heads=attn_cfg.get('num_heads', 8),
                num_regions=attn_cfg.get('num_regions', 4),
                use_flash_attn=attn_cfg.get('use_flash_attn', True)
            )
        
        # Bottleneck blocks
        self.bottlenecks = nn.ModuleList([
            Bottleneck(hidden_channels, hidden_channels, shortcut=True)
            for _ in range(num_blocks)
        ])
        
        # Merge with 1x1 conv
        merge_channels = hidden_channels * (2 + num_blocks)
        if use_area_attn:
            merge_channels += hidden_channels
        self.cv3 = ConvModule(merge_channels, out_channels, 1)
        
        # Residual connection
        self.shortcut = in_channels == out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through R-ELAN."""
        identity = x
        
        # Split
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        
        # Collect outputs for concatenation
        outputs = [y1, y2]
        
        # Area Attention branch
        if self.use_area_attn:
            y_attn = self.area_attn(y2)
            outputs.append(y_attn)
        
        # Transform through bottlenecks
        y = y2
        for block in self.bottlenecks:
            y = block(y)
            outputs.append(y)
        
        # Merge
        out = torch.cat(outputs, dim=1)
        out = self.cv3(out)
        
        # Residual
        if self.shortcut:
            out = out + identity
            
        return out


class SPPFModule(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) module.
    
    Aggregates multi-scale information using cascaded max pooling.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvModule(in_channels, hidden_channels, 1)
        self.cv2 = ConvModule(hidden_channels * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DownsampleBlock(nn.Module):
    """Downsample block with strided convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, 3, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class YOLOv12Backbone(nn.Module):
    """
    YOLOv12 Backbone for visual feature extraction.
    
    Extracts multi-scale features {C3, C4, C5} from input images using
    R-ELAN blocks with Area Attention for efficient long-range modeling.
    
    Architecture:
    - Stage 1: Input -> C1 (stride 2)
    - Stage 2: C1 -> C2 (stride 4)
    - Stage 3: C2 -> C3 (stride 8)
    - Stage 4: C3 -> C4 (stride 16)
    - Stage 5: C4 -> C5 (stride 32) + SPPF
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth_multiple: float = 0.67,
        width_multiple: float = 0.75,
        out_indices: Tuple[int, ...] = (2, 3, 4),
        area_attn_cfg: Optional[dict] = None
    ):
        """
        Initialize YOLOv12 Backbone.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base channel count (scaled by width_multiple)
            depth_multiple: Depth scaling factor
            width_multiple: Width scaling factor
            out_indices: Indices of stages to output features from
            area_attn_cfg: Configuration for Area Attention
        """
        super().__init__()
        
        self.out_indices = out_indices
        
        # Scale channel counts
        def make_divisible(x, divisor=8):
            return int(math.ceil(x / divisor) * divisor)
        
        ch1 = make_divisible(base_channels * width_multiple)
        ch2 = make_divisible(base_channels * 2 * width_multiple)
        ch3 = make_divisible(base_channels * 4 * width_multiple)
        ch4 = make_divisible(base_channels * 8 * width_multiple)
        ch5 = make_divisible(base_channels * 16 * width_multiple)
        
        # Scale block counts
        num_blocks = max(1, int(3 * depth_multiple))
        
        # Stem
        self.stem = ConvModule(in_channels, ch1, 3, stride=2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvModule(ch1, ch2, 3, stride=2),
            RELAN(ch2, ch2, num_blocks=num_blocks, use_area_attn=False)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvModule(ch2, ch3, 3, stride=2),
            RELAN(ch3, ch3, num_blocks=num_blocks, use_area_attn=True, area_attn_cfg=area_attn_cfg)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvModule(ch3, ch4, 3, stride=2),
            RELAN(ch4, ch4, num_blocks=num_blocks, use_area_attn=True, area_attn_cfg=area_attn_cfg)
        )
        
        # Stage 4 (final stage with single R-ELAN for efficiency)
        self.stage4 = nn.Sequential(
            ConvModule(ch4, ch5, 3, stride=2),
            RELAN(ch5, ch5, num_blocks=1, use_area_attn=True, area_attn_cfg=area_attn_cfg),
            SPPFModule(ch5, ch5)
        )
        
        # Output channels for each stage
        self.out_channels = [ch3, ch4, ch5]  # C3, C4, C5
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through backbone.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            List of feature maps [C3, C4, C5] with strides [8, 16, 32]
        """
        outputs = []
        
        x = self.stem(x)      # stride 2
        x = self.stage1(x)    # stride 4
        c3 = self.stage2(x)   # stride 8
        c4 = self.stage3(c3)  # stride 16
        c5 = self.stage4(c4)  # stride 32
        
        # Return features based on out_indices
        features = [c3, c4, c5]
        for i in self.out_indices:
            outputs.append(features[i - 2])  # Adjust for 0-indexed
            
        return outputs


def build_backbone(cfg: dict) -> YOLOv12Backbone:
    """
    Build YOLOv12 backbone from configuration.
    
    Args:
        cfg: Backbone configuration dictionary
        
    Returns:
        Initialized YOLOv12Backbone
    """
    area_attn_cfg = cfg.get('area_attention', {})
    
    return YOLOv12Backbone(
        in_channels=cfg.get('input_channels', 3),
        base_channels=64,
        depth_multiple=cfg.get('depth_multiple', 0.67),
        width_multiple=cfg.get('width_multiple', 0.75),
        out_indices=(2, 3, 4),
        area_attn_cfg=area_attn_cfg
    )
