"""
Shared Model Components

This module provides common building blocks used across the PLVLDet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """Calculate 'same' padding."""
    if padding is None:
        padding = dilation * (kernel_size - 1) // 2
    return padding


class ConvModule(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    
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
    
    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with fused conv-bn."""
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise separable convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride,
            kernel_size // 2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dwconv(x)))
        return self.act(self.bn2(self.pwconv(x)))


class Focus(nn.Module):
    """Focus layer - reduces spatial resolution while increasing channels."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = ConvModule(in_channels * 4, out_channels, kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice and concatenate
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1))


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


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5
    ):
        super().__init__()
        self.hidden = int(out_channels * expansion)
        self.cv1 = ConvModule(in_channels, 2 * self.hidden, 1)
        self.cv2 = ConvModule((2 + num_blocks) * self.hidden, out_channels, 1)
        self.blocks = nn.ModuleList([
            Bottleneck(self.hidden, self.hidden, shortcut, groups, expansion=1.0)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(block(y[-1]) for block in self.blocks)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvModule(in_channels, hidden, 1)
        self.cv2 = ConvModule(hidden * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Concat(nn.Module):
    """Concatenate tensors along specified dimension."""
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: list) -> torch.Tensor:
        return torch.cat(x, dim=self.dim)


class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = self.avg_pool(x).view(B, C)
        max_val = self.max_pool(x).view(B, C)
        attn = self.sigmoid(self.fc(avg) + self.fc(max_val))
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_val = x.max(dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg, max_val], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# Re-export from backbone and PA-VL-PAN modules
from .backbone import RELAN, AreaAttention
from .pa_vl_pan import (
    StripPoolingAttention,
    PTCSPLayer,
    SmallObjectEnhancementModule
)
from .detector import VLDetectionHead

__all__ = [
    'ConvModule',
    'DWConv',
    'Focus',
    'Bottleneck',
    'C2f',
    'SPPF',
    'Concat',
    'ChannelAttention',
    'SpatialAttention',
    'CBAM',
    'RELAN',
    'AreaAttention',
    'StripPoolingAttention',
    'PTCSPLayer',
    'SmallObjectEnhancementModule',
    'VLDetectionHead'
]
