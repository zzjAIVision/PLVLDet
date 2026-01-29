#!/usr/bin/env python3
"""
LabelMe Annotation Converter

Converts LabelMe JSON annotation files to PLVLDet internal format.
Supports both rectangle and polygon shape types.

Usage:
    python tools/labelme_converter.py \
        --image_dir data_images \
        --label_dir data_labels \
        --output_dir data/annotations \
        --format coco
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm


CATEGORY_MAPPING = {
    # Full names
    'Insulator': 0,
    'Ceramic insulator': 1,
    'Glass insulator': 2,
    'Polymer insulator': 3,
    'Normal glass insulator': 4,
    'Broken glass insulator': 5,
    'Contaminated glass insulator': 6,
    'Normal Polymer insulator': 7,
    'Damaged Polymer insulator': 8,
    'Contaminated Polymer insulator': 9,
    'Deformed Polymer insulator': 10,
    'Normal ceramic insulator': 11,
    'Contaminated ceramic insulator': 12,
    'Broken ceramic insulator': 13,
    'Fall-off ceramic insulator': 14,
    'Foreign matters on tower': 15,
    "Bird's nest on tower": 16,
    'Plastic bag on tower': 17,
    'Kite on tower': 18,
    'Tower balloon': 19,
    'Wire foreign matter': 20,
    'Wire plastic bag': 21,
    'Wire kite': 22,
    'Wire balloon': 23,
    'Broken strand of transmission line': 24,
    'Loose strand of transmission line': 25,
    'vibration damper': 26,
    'Displacement of damper': 27,
    'Missing damper': 28,
    'Tension clamp': 29,
    'armour clamp': 30,
    'Missing split pin': 31,
    # Abbreviations
    'IT': 0, 'CI': 1, 'GI': 2, 'PI': 3,
    'NGI': 4, 'BGI': 5, 'CGI': 6,
    'NPI': 7, 'DaPI': 8, 'CPI': 9, 'DePI': 10,
    'NCI': 11, 'CCI': 12, 'BCI': 13, 'FoCI': 14,
    'FMT': 15, 'BNT': 16, 'PBT': 17, 'KoT': 18, 'TB': 19,
    'WFM': 20, 'WPB': 21, 'WK': 22, 'WB': 23,
    'BSTL': 24, 'LSTL': 25,
    'VD': 26, 'DD': 27, 'MD': 28,
    'TC': 29, 'AC': 30, 'MSP': 31
}

CATEGORY_NAMES = [
    'Insulator', 'Ceramic insulator', 'Glass insulator', 'Polymer insulator',
    'Normal glass insulator', 'Broken glass insulator', 'Contaminated glass insulator',
    'Normal Polymer insulator', 'Damaged Polymer insulator', 'Contaminated Polymer insulator',
    'Deformed Polymer insulator', 'Normal ceramic insulator', 'Contaminated ceramic insulator',
    'Broken ceramic insulator', 'Fall-off ceramic insulator', 'Foreign matters on tower',
    "Bird's nest on tower", 'Plastic bag on tower', 'Kite on tower', 'Tower balloon',
    'Wire foreign matter', 'Wire plastic bag', 'Wire kite', 'Wire balloon',
    'Broken strand of transmission line', 'Loose strand of transmission line',
    'vibration damper', 'Displacement of damper', 'Missing damper',
    'Tension clamp', 'armour clamp', 'Missing split pin'
]


class LabelMeConverter:
    """Converts LabelMe annotations to various formats."""
    
    def __init__(self, image_dir, label_dir, output_dir, format_type='coco'):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.annotations = []
        self.images = []
        self.category_stats = defaultdict(int)
        
    def parse_labelme_json(self, json_path):
        """Parse a single LabelMe JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        image_path = data.get('imagePath', '')
        image_height = data.get('imageHeight', 0)
        image_width = data.get('imageWidth', 0)
        
        annotations = []
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            shape_type = shape.get('shape_type', '')
            points = shape.get('points', [])
            
            # Skip if label not in mapping
            if label not in CATEGORY_MAPPING:
                print(f"Warning: Unknown label '{label}' in {json_path}")
                continue
                
            category_id = CATEGORY_MAPPING[label]
            
            # Convert points to bbox
            if shape_type == 'rectangle':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    bbox = [min(x1, x2), min(y1, y2), 
                            abs(x2 - x1), abs(y2 - y1)]
            elif shape_type == 'polygon':
                points_arr = np.array(points)
                x_min, y_min = points_arr.min(axis=0)
                x_max, y_max = points_arr.max(axis=0)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                continue
                
            # Validate bbox
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
                
            annotations.append({
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'segmentation': [np.array(points).flatten().tolist()] if shape_type == 'polygon' else [],
                'iscrowd': 0
            })
            
            self.category_stats[category_id] += 1
            
        return {
            'image_path': image_path,
            'height': image_height,
            'width': image_width,
            'annotations': annotations
        }
        
    def convert_to_coco(self):
        """Convert all annotations to COCO format."""
        coco_output = {
            'info': {
                'description': 'PLVLDet Power Line Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'PLVLDet Team',
                'date_created': datetime.now().strftime('%Y-%m-%d')
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for idx, name in enumerate(CATEGORY_NAMES):
            coco_output['categories'].append({
                'id': idx,
                'name': name,
                'supercategory': self._get_supercategory(idx)
            })
            
        # Process all JSON files
        json_files = list(self.label_dir.glob('*.json'))
        
        image_id = 0
        annotation_id = 0
        
        for json_path in tqdm(json_files, desc='Converting annotations'):
            parsed = self.parse_labelme_json(json_path)
            
            if not parsed['annotations']:
                continue
                
            # Find corresponding image
            image_name = Path(parsed['image_path']).name
            image_path = self.image_dir / image_name
            
            if not image_path.exists():
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    alt_path = self.image_dir / (json_path.stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break
                        
            # Get image dimensions if not in JSON
            height, width = parsed['height'], parsed['width']
            if height == 0 or width == 0:
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        height, width = img.shape[:2]
                        
            if height == 0 or width == 0:
                print(f"Warning: Cannot determine image size for {json_path}")
                continue
                
            # Add image entry
            coco_output['images'].append({
                'id': image_id,
                'file_name': image_path.name,
                'height': height,
                'width': width
            })
            
            # Add annotations
            for ann in parsed['annotations']:
                coco_output['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'segmentation': ann['segmentation'],
                    'iscrowd': ann['iscrowd']
                })
                annotation_id += 1
                
            image_id += 1
            
        # Save output
        output_path = self.output_dir / 'annotations.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, indent=2)
            
        print(f"\nConverted {image_id} images with {annotation_id} annotations")
        print(f"Output saved to: {output_path}")
        
        return coco_output
        
    def convert_to_yolo(self):
        """Convert all annotations to YOLO format."""
        labels_output_dir = self.output_dir / 'labels'
        labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = list(self.label_dir.glob('*.json'))
        
        processed = 0
        for json_path in tqdm(json_files, desc='Converting to YOLO format'):
            parsed = self.parse_labelme_json(json_path)
            
            if not parsed['annotations']:
                continue
                
            # Get image dimensions
            height, width = parsed['height'], parsed['width']
            
            if height == 0 or width == 0:
                image_name = Path(parsed['image_path']).name
                image_path = self.image_dir / image_name
                if not image_path.exists():
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        alt_path = self.image_dir / (json_path.stem + ext)
                        if alt_path.exists():
                            image_path = alt_path
                            break
                            
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        height, width = img.shape[:2]
                        
            if height == 0 or width == 0:
                continue
                
            # Write YOLO format labels
            output_path = labels_output_dir / (json_path.stem + '.txt')
            with open(output_path, 'w') as f:
                for ann in parsed['annotations']:
                    bbox = ann['bbox']
                    # Convert to YOLO format (center_x, center_y, w, h) normalized
                    cx = (bbox[0] + bbox[2] / 2) / width
                    cy = (bbox[1] + bbox[3] / 2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height
                    
                    f.write(f"{ann['category_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    
            processed += 1
            
        print(f"\nConverted {processed} annotation files to YOLO format")
        print(f"Output saved to: {labels_output_dir}")
        
    def convert_to_internal(self):
        """Convert to PLVLDet internal format."""
        output_file = self.output_dir / 'dataset.json'
        
        json_files = list(self.label_dir.glob('*.json'))
        
        dataset = {
            'images': [],
            'categories': CATEGORY_NAMES,
            'statistics': {}
        }
        
        for json_path in tqdm(json_files, desc='Converting to internal format'):
            parsed = self.parse_labelme_json(json_path)
            
            if not parsed['annotations']:
                continue
                
            # Get image path
            image_name = Path(parsed['image_path']).name
            image_path = self.image_dir / image_name
            
            if not image_path.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    alt_path = self.image_dir / (json_path.stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        image_name = alt_path.name
                        break
                        
            height, width = parsed['height'], parsed['width']
            
            if height == 0 or width == 0:
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        height, width = img.shape[:2]
                        
            if height == 0 or width == 0:
                continue
                
            entry = {
                'file_name': image_name,
                'height': height,
                'width': width,
                'annotations': []
            }
            
            for ann in parsed['annotations']:
                entry['annotations'].append({
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox']  # [x, y, w, h]
                })
                
            dataset['images'].append(entry)
            
        # Calculate statistics
        total_annotations = sum(self.category_stats.values())
        dataset['statistics'] = {
            'num_images': len(dataset['images']),
            'num_annotations': total_annotations,
            'category_distribution': dict(self.category_stats)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
            
        print(f"\nConverted {len(dataset['images'])} images")
        print(f"Total annotations: {total_annotations}")
        print(f"Output saved to: {output_file}")
        
        return dataset
        
    def _get_supercategory(self, category_id):
        """Get supercategory for hierarchical structure."""
        if category_id == 0:
            return 'root'
        elif category_id in [1, 2, 3]:
            return 'Insulator'
        elif category_id in [4, 5, 6]:
            return 'Glass insulator'
        elif category_id in [7, 8, 9, 10]:
            return 'Polymer insulator'
        elif category_id in [11, 12, 13, 14]:
            return 'Ceramic insulator'
        elif category_id == 15:
            return 'root'
        elif category_id in [16, 17, 18, 19]:
            return 'Foreign matters on tower'
        elif category_id == 20:
            return 'root'
        elif category_id in [21, 22, 23]:
            return 'Wire foreign matter'
        elif category_id in [24, 25]:
            return 'Wire'
        elif category_id == 26:
            return 'root'
        elif category_id in [27, 28]:
            return 'vibration damper'
        elif category_id in [29, 30]:
            return 'Clamp'
        else:
            return 'root'
            
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        
        total = sum(self.category_stats.values())
        print(f"Total annotations: {total}")
        print("\nPer-category distribution:")
        
        for cat_id in sorted(self.category_stats.keys()):
            name = CATEGORY_NAMES[cat_id]
            count = self.category_stats[cat_id]
            pct = count / total * 100 if total > 0 else 0
            print(f"  [{cat_id:2d}] {name:40s}: {count:6d} ({pct:5.2f}%)")
            
    def convert(self):
        """Run conversion based on format type."""
        if self.format_type == 'coco':
            self.convert_to_coco()
        elif self.format_type == 'yolo':
            self.convert_to_yolo()
        elif self.format_type == 'internal':
            self.convert_to_internal()
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")
            
        self.print_statistics()


def main():
    parser = argparse.ArgumentParser(
        description='Convert LabelMe annotations to various formats'
    )
    parser.add_argument(
        '--image_dir', type=str, default='data_images',
        help='Directory containing images'
    )
    parser.add_argument(
        '--label_dir', type=str, default='data_labels',
        help='Directory containing LabelMe JSON files'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/annotations',
        help='Output directory for converted annotations'
    )
    parser.add_argument(
        '--format', type=str, default='coco',
        choices=['coco', 'yolo', 'internal'],
        help='Output annotation format'
    )
    
    args = parser.parse_args()
    
    converter = LabelMeConverter(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        format_type=args.format
    )
    
    converter.convert()


if __name__ == '__main__':
    main()
