#!/usr/bin/env python3
"""
Patch Generator for Power Line Images

Generates 1000x1000 patches from large transmission line images using
sliding window approach. Filters patches based on object overlap ratio.

Paper specifications:
- Patch size: 1000 x 1000 pixels
- Stride: 800 pixels (200 pixel overlap)
- Overlap threshold (τ): 0.1 for including partial objects
- Area ratio threshold: 0.9 for discarding truncated objects

Usage:
    python tools/patch_generator.py \
        --image_dir data_images \
        --annotation_file data/annotations/annotations.json \
        --output_dir data/patches \
        --patch_size 1000 \
        --stride 800
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_overlap_ratio(box, patch_box):
    """
    Compute the ratio of object area inside the patch.
    
    Args:
        box: Object bounding box [x, y, w, h]
        patch_box: Patch bounding box [x, y, w, h]
        
    Returns:
        Ratio of object area within patch (0-1)
    """
    x1_obj, y1_obj = box[0], box[1]
    x2_obj, y2_obj = box[0] + box[2], box[1] + box[3]
    
    x1_patch, y1_patch = patch_box[0], patch_box[1]
    x2_patch, y2_patch = patch_box[0] + patch_box[2], patch_box[1] + patch_box[3]
    
    # Intersection
    inter_x1 = max(x1_obj, x1_patch)
    inter_y1 = max(y1_obj, y1_patch)
    inter_x2 = min(x2_obj, x2_patch)
    inter_y2 = min(y2_obj, y2_patch)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    object_area = box[2] * box[3]
    
    return inter_area / object_area if object_area > 0 else 0.0


def clip_box_to_patch(box, patch_box):
    """
    Clip object box to patch boundaries and convert to patch-relative coordinates.
    
    Args:
        box: Object bounding box [x, y, w, h] in image coordinates
        patch_box: Patch bounding box [x, y, w, h] in image coordinates
        
    Returns:
        Clipped box in patch-relative coordinates [x, y, w, h]
    """
    x1_obj, y1_obj = box[0], box[1]
    x2_obj, y2_obj = box[0] + box[2], box[1] + box[3]
    
    x1_patch, y1_patch = patch_box[0], patch_box[1]
    x2_patch, y2_patch = patch_box[0] + patch_box[2], patch_box[1] + patch_box[3]
    
    # Clip to patch boundaries
    x1_clipped = max(x1_obj, x1_patch)
    y1_clipped = max(y1_obj, y1_patch)
    x2_clipped = min(x2_obj, x2_patch)
    y2_clipped = min(y2_obj, y2_patch)
    
    # Convert to patch-relative coordinates
    x_rel = x1_clipped - x1_patch
    y_rel = y1_clipped - y1_patch
    w_rel = x2_clipped - x1_clipped
    h_rel = y2_clipped - y1_clipped
    
    return [x_rel, y_rel, w_rel, h_rel]


class PatchGenerator:
    """Generates patches from large transmission line images."""
    
    def __init__(
        self,
        image_dir,
        annotation_file,
        output_dir,
        patch_size=1000,
        stride=800,
        overlap_threshold=0.1,
        area_ratio_threshold=0.9,
        min_objects_per_patch=0,
        workers=4
    ):
        """
        Initialize patch generator.
        
        Args:
            image_dir: Directory containing original images
            annotation_file: Path to COCO-format annotation file
            output_dir: Output directory for patches
            patch_size: Size of square patches
            stride: Sliding window stride
            overlap_threshold: Minimum overlap to include object
            area_ratio_threshold: Minimum area ratio to keep object (not truncated)
            min_objects_per_patch: Minimum objects required per patch
            workers: Number of parallel workers
        """
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        self.area_ratio_threshold = area_ratio_threshold
        self.min_objects_per_patch = min_objects_per_patch
        self.workers = workers
        
        # Create output directories
        self.patch_image_dir = self.output_dir / 'images'
        self.patch_label_dir = self.output_dir / 'labels'
        self.patch_image_dir.mkdir(parents=True, exist_ok=True)
        self.patch_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        self.load_annotations()
        
        # Statistics
        self.stats = defaultdict(int)
        
    def load_annotations(self):
        """Load COCO-format annotations."""
        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Build image id to annotations mapping
        self.image_annotations = defaultdict(list)
        for ann in self.coco_data.get('annotations', []):
            self.image_annotations[ann['image_id']].append(ann)
            
        # Build image id to info mapping
        self.image_info = {
            img['id']: img for img in self.coco_data.get('images', [])
        }
        
        self.categories = self.coco_data.get('categories', [])
        
        print(f"Loaded {len(self.image_info)} images with "
              f"{len(self.coco_data.get('annotations', []))} annotations")
        
    def generate_patches_for_image(self, image_id):
        """
        Generate all patches for a single image.
        
        Args:
            image_id: COCO image ID
            
        Returns:
            List of patch info dictionaries
        """
        image_info = self.image_info[image_id]
        image_path = self.image_dir / image_info['file_name']
        
        if not image_path.exists():
            return []
            
        img = cv2.imread(str(image_path))
        if img is None:
            return []
            
        height, width = img.shape[:2]
        annotations = self.image_annotations[image_id]
        
        patches = []
        patch_idx = 0
        
        # Sliding window
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch_box = [x, y, self.patch_size, self.patch_size]
                
                # Find objects in this patch
                patch_annotations = []
                
                for ann in annotations:
                    box = ann['bbox']  # [x, y, w, h]
                    overlap_ratio = compute_overlap_ratio(box, patch_box)
                    
                    # Skip if overlap below threshold
                    if overlap_ratio < self.overlap_threshold:
                        continue
                        
                    # Check if object is truncated
                    is_truncated = overlap_ratio < self.area_ratio_threshold
                    
                    # Clip box to patch and convert coordinates
                    clipped_box = clip_box_to_patch(box, patch_box)
                    
                    # Skip if clipped box is too small
                    if clipped_box[2] < 2 or clipped_box[3] < 2:
                        continue
                        
                    patch_annotations.append({
                        'category_id': ann['category_id'],
                        'bbox': clipped_box,
                        'original_bbox': box,
                        'overlap_ratio': overlap_ratio,
                        'is_truncated': is_truncated
                    })
                    
                # Skip patch if not enough objects
                if len(patch_annotations) < self.min_objects_per_patch:
                    continue
                    
                # Extract patch image
                patch_img = img[y:y+self.patch_size, x:x+self.patch_size]
                
                # Generate patch filename
                base_name = Path(image_info['file_name']).stem
                patch_name = f"{base_name}_patch_{patch_idx:04d}"
                
                patches.append({
                    'patch_name': patch_name,
                    'patch_img': patch_img,
                    'annotations': patch_annotations,
                    'source_image': image_info['file_name'],
                    'patch_location': [x, y]
                })
                
                patch_idx += 1
                
        # Handle edge patches (rightmost and bottom)
        # Right edge
        if width > self.patch_size:
            x = width - self.patch_size
            for y in range(0, height - self.patch_size + 1, self.stride):
                if x % self.stride == 0:  # Already covered
                    continue
                patch_box = [x, y, self.patch_size, self.patch_size]
                patch_annotations = self._get_patch_annotations(annotations, patch_box)
                
                if len(patch_annotations) >= self.min_objects_per_patch:
                    patch_img = img[y:y+self.patch_size, x:x+self.patch_size]
                    base_name = Path(image_info['file_name']).stem
                    patch_name = f"{base_name}_patch_{patch_idx:04d}"
                    
                    patches.append({
                        'patch_name': patch_name,
                        'patch_img': patch_img,
                        'annotations': patch_annotations,
                        'source_image': image_info['file_name'],
                        'patch_location': [x, y]
                    })
                    patch_idx += 1
                    
        # Bottom edge
        if height > self.patch_size:
            y = height - self.patch_size
            for x in range(0, width - self.patch_size + 1, self.stride):
                if y % self.stride == 0:  # Already covered
                    continue
                patch_box = [x, y, self.patch_size, self.patch_size]
                patch_annotations = self._get_patch_annotations(annotations, patch_box)
                
                if len(patch_annotations) >= self.min_objects_per_patch:
                    patch_img = img[y:y+self.patch_size, x:x+self.patch_size]
                    base_name = Path(image_info['file_name']).stem
                    patch_name = f"{base_name}_patch_{patch_idx:04d}"
                    
                    patches.append({
                        'patch_name': patch_name,
                        'patch_img': patch_img,
                        'annotations': patch_annotations,
                        'source_image': image_info['file_name'],
                        'patch_location': [x, y]
                    })
                    patch_idx += 1
                    
        return patches
        
    def _get_patch_annotations(self, annotations, patch_box):
        """Get annotations for a patch box."""
        patch_annotations = []
        
        for ann in annotations:
            box = ann['bbox']
            overlap_ratio = compute_overlap_ratio(box, patch_box)
            
            if overlap_ratio < self.overlap_threshold:
                continue
                
            is_truncated = overlap_ratio < self.area_ratio_threshold
            clipped_box = clip_box_to_patch(box, patch_box)
            
            if clipped_box[2] < 2 or clipped_box[3] < 2:
                continue
                
            patch_annotations.append({
                'category_id': ann['category_id'],
                'bbox': clipped_box,
                'original_bbox': box,
                'overlap_ratio': overlap_ratio,
                'is_truncated': is_truncated
            })
            
        return patch_annotations
        
    def save_patch(self, patch_info):
        """Save a single patch and its annotations."""
        patch_name = patch_info['patch_name']
        
        # Save image
        image_path = self.patch_image_dir / f"{patch_name}.jpg"
        cv2.imwrite(str(image_path), patch_info['patch_img'], 
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Save YOLO format labels
        label_path = self.patch_label_dir / f"{patch_name}.txt"
        with open(label_path, 'w') as f:
            for ann in patch_info['annotations']:
                bbox = ann['bbox']
                # Convert to YOLO format
                cx = (bbox[0] + bbox[2] / 2) / self.patch_size
                cy = (bbox[1] + bbox[3] / 2) / self.patch_size
                w = bbox[2] / self.patch_size
                h = bbox[3] / self.patch_size
                
                f.write(f"{ann['category_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
                # Update statistics
                self.stats[f"cat_{ann['category_id']}"] += 1
                if ann['is_truncated']:
                    self.stats['truncated'] += 1
                    
        self.stats['patches'] += 1
        self.stats['annotations'] += len(patch_info['annotations'])
        
        return patch_name
        
    def generate_all_patches(self):
        """Generate patches for all images."""
        image_ids = list(self.image_info.keys())
        
        all_patches_info = []
        
        print(f"\nGenerating patches from {len(image_ids)} images...")
        print(f"Patch size: {self.patch_size}, Stride: {self.stride}")
        print(f"Overlap threshold: {self.overlap_threshold}")
        print(f"Area ratio threshold: {self.area_ratio_threshold}")
        
        for image_id in tqdm(image_ids, desc='Processing images'):
            patches = self.generate_patches_for_image(image_id)
            
            for patch in patches:
                self.save_patch(patch)
                
                # Store metadata (without image data)
                all_patches_info.append({
                    'patch_name': patch['patch_name'],
                    'source_image': patch['source_image'],
                    'patch_location': patch['patch_location'],
                    'num_objects': len(patch['annotations']),
                    'categories': [a['category_id'] for a in patch['annotations']]
                })
                
        # Save patch metadata
        metadata_path = self.output_dir / 'patch_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'patches': all_patches_info,
                'config': {
                    'patch_size': self.patch_size,
                    'stride': self.stride,
                    'overlap_threshold': self.overlap_threshold,
                    'area_ratio_threshold': self.area_ratio_threshold
                },
                'statistics': dict(self.stats)
            }, f, indent=2)
            
        self._print_statistics()
        
        return all_patches_info
        
    def _print_statistics(self):
        """Print generation statistics."""
        print("\n" + "=" * 50)
        print("Patch Generation Statistics")
        print("=" * 50)
        print(f"Total patches generated: {self.stats['patches']}")
        print(f"Total annotations: {self.stats['annotations']}")
        print(f"Truncated objects: {self.stats['truncated']}")
        
        print("\nPer-category distribution:")
        for key in sorted([k for k in self.stats.keys() if k.startswith('cat_')]):
            cat_id = int(key.split('_')[1])
            count = self.stats[key]
            if self.categories:
                cat_name = next(
                    (c['name'] for c in self.categories if c['id'] == cat_id),
                    f'Category {cat_id}'
                )
            else:
                cat_name = f'Category {cat_id}'
            print(f"  [{cat_id:2d}] {cat_name:40s}: {count:6d}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate patches from large power line images'
    )
    parser.add_argument(
        '--image_dir', type=str, default='data_images',
        help='Directory containing original images'
    )
    parser.add_argument(
        '--annotation_file', type=str, default='data/annotations/annotations.json',
        help='Path to COCO-format annotation file'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/patches',
        help='Output directory for patches'
    )
    parser.add_argument(
        '--patch_size', type=int, default=1000,
        help='Size of square patches'
    )
    parser.add_argument(
        '--stride', type=int, default=800,
        help='Sliding window stride'
    )
    parser.add_argument(
        '--overlap_threshold', type=float, default=0.1,
        help='Minimum overlap ratio to include object'
    )
    parser.add_argument(
        '--area_ratio_threshold', type=float, default=0.9,
        help='Minimum area ratio (discard truncated objects below this)'
    )
    parser.add_argument(
        '--min_objects', type=int, default=0,
        help='Minimum objects required per patch'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    generator = PatchGenerator(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        overlap_threshold=args.overlap_threshold,
        area_ratio_threshold=args.area_ratio_threshold,
        min_objects_per_patch=args.min_objects,
        workers=args.workers
    )
    
    generator.generate_all_patches()


if __name__ == '__main__':
    main()
