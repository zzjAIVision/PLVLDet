"""
Data augmentation transforms for PLVLDet.
Implements augmentation pipeline including HSV jitter, affine transforms,
random flip, mosaic and mixup augmentations.
"""

import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Any


class HSVAugment:
    """HSV color space augmentation."""
    
    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.h_gain == 0 and self.s_gain == 0 and self.v_gain == 0:
            return image
        
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class RandomAffine:
    """Random affine transformation including rotation, scale, shear, and translation."""
    
    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        border_value: Tuple[int, int, int] = (114, 114, 114)
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border_value = border_value
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        img_size: Tuple[int, int] = (1000, 1000)
    ) -> Tuple[np.ndarray, np.ndarray]:
        height, width = img_size
        
        # Center
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2
        
        # Rotation and scale
        R = np.eye(3)
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)
        
        # Shear
        S = np.eye(3)
        S[0, 1] = np.tan(random.uniform(-self.shear, self.shear) * np.pi / 180)
        S[1, 0] = np.tan(random.uniform(-self.shear, self.shear) * np.pi / 180)
        
        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height
        
        # Combined transformation matrix
        M = T @ S @ R @ C
        
        # Transform image
        img_transformed = cv2.warpAffine(
            image,
            M[:2],
            dsize=(width, height),
            borderValue=self.border_value
        )
        
        # Transform boxes
        if len(boxes) > 0:
            boxes_transformed = self._transform_boxes(boxes, M, width, height)
        else:
            boxes_transformed = boxes
        
        return img_transformed, boxes_transformed
    
    def _transform_boxes(
        self,
        boxes: np.ndarray,
        M: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """Transform bounding boxes using affine matrix."""
        n = len(boxes)
        if n == 0:
            return boxes
        
        # Get box corners
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)
        
        # Get new box coordinates
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_boxes = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))
        ).reshape(4, n).T
        
        # Clip boxes
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, width)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, height)
        
        return new_boxes


class Mosaic:
    """4-image mosaic augmentation."""
    
    def __init__(
        self,
        img_size: int = 1000,
        mosaic_prob: float = 0.5,
        border_value: Tuple[int, int, int] = (114, 114, 114)
    ):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.border_value = border_value
    
    def __call__(
        self,
        images: List[np.ndarray],
        boxes_list: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation to 4 images.
        
        Args:
            images: List of 4 images
            boxes_list: List of 4 box arrays (xyxy format)
            labels_list: List of 4 label arrays
            
        Returns:
            mosaic_img: Combined image
            mosaic_boxes: Combined boxes
            mosaic_labels: Combined labels
        """
        assert len(images) == 4
        
        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)
        
        mosaic_img = np.full((s * 2, s * 2, 3), self.border_value, dtype=np.uint8)
        all_boxes = []
        all_labels = []
        
        for i, (img, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
            h, w = img.shape[:2]
            
            # Place image in mosaic
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust boxes
            padw = x1a - x1b
            padh = y1a - y1b
            
            if len(boxes) > 0:
                boxes_adjusted = boxes.copy()
                boxes_adjusted[:, [0, 2]] += padw
                boxes_adjusted[:, [1, 3]] += padh
                all_boxes.append(boxes_adjusted)
                all_labels.append(labels)
        
        if len(all_boxes) > 0:
            mosaic_boxes = np.concatenate(all_boxes, axis=0)
            mosaic_labels = np.concatenate(all_labels, axis=0)
            
            # Clip boxes
            mosaic_boxes[:, [0, 2]] = mosaic_boxes[:, [0, 2]].clip(0, 2 * s)
            mosaic_boxes[:, [1, 3]] = mosaic_boxes[:, [1, 3]].clip(0, 2 * s)
            
            # Filter valid boxes
            valid_mask = self._filter_valid_boxes(mosaic_boxes)
            mosaic_boxes = mosaic_boxes[valid_mask]
            mosaic_labels = mosaic_labels[valid_mask]
        else:
            mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.zeros((0,), dtype=np.int64)
        
        return mosaic_img, mosaic_boxes, mosaic_labels
    
    def _filter_valid_boxes(self, boxes: np.ndarray, min_area: float = 16.0) -> np.ndarray:
        """Filter boxes with minimum area."""
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        return areas >= min_area


class MixUp:
    """MixUp augmentation for object detection."""
    
    def __init__(self, alpha: float = 0.5, mixup_prob: float = 0.1):
        self.alpha = alpha
        self.mixup_prob = mixup_prob
    
    def __call__(
        self,
        image1: np.ndarray,
        boxes1: np.ndarray,
        labels1: np.ndarray,
        image2: np.ndarray,
        boxes2: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation to two images.
        """
        if random.random() > self.mixup_prob:
            return image1, boxes1, labels1
        
        # Sample mixing ratio
        ratio = np.random.beta(self.alpha, self.alpha)
        ratio = max(ratio, 1 - ratio)  # Ensure ratio >= 0.5
        
        # Mix images
        mixed_img = (image1.astype(np.float32) * ratio + 
                    image2.astype(np.float32) * (1 - ratio)).astype(np.uint8)
        
        # Combine boxes and labels
        if len(boxes1) > 0 and len(boxes2) > 0:
            mixed_boxes = np.concatenate([boxes1, boxes2], axis=0)
            mixed_labels = np.concatenate([labels1, labels2], axis=0)
        elif len(boxes1) > 0:
            mixed_boxes = boxes1
            mixed_labels = labels1
        elif len(boxes2) > 0:
            mixed_boxes = boxes2
            mixed_labels = labels2
        else:
            mixed_boxes = np.zeros((0, 4), dtype=np.float32)
            mixed_labels = np.zeros((0,), dtype=np.int64)
        
        return mixed_img, mixed_boxes, mixed_labels


class LetterBox:
    """Resize and pad image while preserving aspect ratio."""
    
    def __init__(
        self,
        new_shape: Tuple[int, int] = (1000, 1000),
        color: Tuple[int, int, int] = (114, 114, 114),
        scaleup: bool = True
    ):
        self.new_shape = new_shape
        self.color = color
        self.scaleup = scaleup
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, Tuple[float, float]]]:
        """
        Apply letterbox transformation.
        
        Returns:
            image: Transformed image
            boxes: Transformed boxes (if provided)
            (ratio, (dw, dh)): Scale ratio and padding
        """
        shape = image.shape[:2]  # H, W
        new_shape = self.new_shape
        
        # Compute scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.color
        )
        
        # Transform boxes
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + dw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + dh
        
        return image, boxes, (r, (dw, dh))


def build_train_transforms(cfg: Dict[str, Any]) -> A.Compose:
    """
    Build training augmentation pipeline.
    
    Args:
        cfg: Configuration dictionary with augmentation parameters
        
    Returns:
        Albumentations compose object
    """
    img_size = cfg.get('img_size', 1000)
    
    transforms = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))
    
    return transforms


def build_val_transforms(cfg: Dict[str, Any]) -> A.Compose:
    """
    Build validation/test augmentation pipeline.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Albumentations compose object
    """
    transforms = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.0
    ))
    
    return transforms


class PLVLDetTransform:
    """
    Complete transform pipeline for PLVLDet.
    Combines custom augmentations with Albumentations.
    """
    
    def __init__(
        self,
        img_size: int = 1000,
        is_train: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        flipud: float = 0.0,
        fliplr: float = 0.5
    ):
        self.img_size = img_size
        self.is_train = is_train
        
        if is_train:
            self.hsv_augment = HSVAugment(hsv_h, hsv_s, hsv_v)
            self.affine = RandomAffine(degrees, translate, scale, shear)
            self.fliplr = fliplr
            self.flipud = flipud
        
        self.letterbox = LetterBox(new_shape=(img_size, img_size))
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply transforms to image and annotations.
        
        Args:
            image: Input image (BGR)
            boxes: Bounding boxes in xyxy format
            labels: Class labels
            
        Returns:
            Transformed image, boxes, and labels
        """
        if self.is_train:
            # HSV augmentation
            image = self.hsv_augment(image)
            
            # Random affine
            image, boxes = self.affine(image, boxes, (self.img_size, self.img_size))
            
            # Random horizontal flip
            if random.random() < self.fliplr:
                image = np.fliplr(image).copy()
                if len(boxes) > 0:
                    boxes[:, [0, 2]] = self.img_size - boxes[:, [2, 0]]
            
            # Random vertical flip
            if random.random() < self.flipud:
                image = np.flipud(image).copy()
                if len(boxes) > 0:
                    boxes[:, [1, 3]] = self.img_size - boxes[:, [3, 1]]
        
        # Letterbox resize
        image, boxes, _ = self.letterbox(image, boxes)
        
        # Filter valid boxes
        if len(boxes) > 0:
            valid_mask = self._filter_boxes(boxes)
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        return image, boxes, labels
    
    def _filter_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Filter boxes with minimum width/height."""
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        return (widths >= 4) & (heights >= 4)
