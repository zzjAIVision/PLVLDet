#!/usr/bin/env python3
"""
PLVLDet Inference Script

Performs object detection on single images or directories of images
using a trained PLVLDet model.

Features:
- Single image inference
- Batch inference on image directories
- Support for custom category prompts
- Configurable confidence and IoU thresholds
- Optional visualization output
"""

import os
import sys
import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.detector import PLVLDet
from models.powerbert import PowerBERT
from data.transforms import get_val_transforms
from utils.postprocess import non_max_suppression, scale_boxes
from utils.visualization import draw_detections


class PLVLDetPredictor:
    """PLVLDet inference wrapper for easy deployment."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        category_file: str,
        device: str = 'cuda',
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.categories = self._load_categories(category_file)
        self.num_classes = len(self.categories)
        
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        self.img_size = self.config['data'].get('img_size', 640)
        self.transforms = get_val_transforms(self.img_size)
        
        self.text_embeddings = self._compute_text_embeddings()
        
        print(f"Model loaded on {self.device}")
        print(f"Number of categories: {self.num_classes}")
        print(f"Confidence threshold: {self.conf_thresh}")
        print(f"IoU threshold: {self.iou_thresh}")
    
    def _load_categories(self, category_file: str) -> Dict:
        """Load category definitions."""
        with open(category_file, 'r') as f:
            cat_config = yaml.safe_load(f)
        
        categories = {}
        for cat_id, cat_info in cat_config.get('categories', {}).items():
            if isinstance(cat_info, dict):
                categories[int(cat_id)] = {
                    'name': cat_info.get('name', f'class_{cat_id}'),
                    'abbreviation': cat_info.get('abbreviation', ''),
                    'prompt': cat_info.get('prompt', cat_info.get('name', ''))
                }
            elif isinstance(cat_info, str):
                categories[int(cat_id)] = {
                    'name': cat_info,
                    'abbreviation': '',
                    'prompt': cat_info
                }
        
        return categories
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load trained PLVLDet model."""
        model = PLVLDet(
            num_classes=self.num_classes,
            backbone_type=self.config['model'].get('backbone_type', 'yolov12n'),
            text_dim=self.config['model'].get('text_dim', 512),
            visual_dim=self.config['model'].get('visual_dim', 256),
            fusion_dim=self.config['model'].get('fusion_dim', 256),
            num_heads=self.config['model'].get('num_heads', 8),
            use_pa_vl_pan=True,
            use_spa=True
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def _compute_text_embeddings(self) -> torch.Tensor:
        """Pre-compute text embeddings for all categories."""
        prompts = [self.categories[i]['prompt'] for i in range(self.num_classes)]
        
        text_embeddings = self.model.encode_text(prompts)
        text_embeddings = text_embeddings.to(self.device)
        
        return text_embeddings
    
    def preprocess_image(
        self, 
        image: Union[str, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float]]:
        """Preprocess image for inference."""
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
            if img.shape[2] == 4:
                img = img[:, :, :3]
        
        original_shape = img.shape[:2]
        
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        img_tensor = torch.from_numpy(img_padded).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        scale_factors = (scale, scale)
        pad_offsets = (left, top)
        
        return img_tensor.to(self.device), original_shape, (scale_factors, pad_offsets)
    
    def postprocess_predictions(
        self,
        predictions: torch.Tensor,
        original_shape: Tuple[int, int],
        preprocess_info: Tuple
    ) -> List[Dict]:
        """Convert raw predictions to detection results."""
        scale_factors, pad_offsets = preprocess_info
        scale_x, scale_y = scale_factors
        pad_x, pad_y = pad_offsets
        
        preds = non_max_suppression(
            predictions,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh
        )
        
        results = []
        
        if preds is not None and len(preds) > 0:
            for pred in preds[0]:
                if len(pred) >= 6:
                    x1, y1, x2, y2, conf, cls_id = pred[:6]
                    
                    x1 = (x1 - pad_x) / scale_x
                    y1 = (y1 - pad_y) / scale_y
                    x2 = (x2 - pad_x) / scale_x
                    y2 = (y2 - pad_y) / scale_y
                    
                    h, w = original_shape
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    
                    cls_id = int(cls_id)
                    category_info = self.categories.get(cls_id, {})
                    
                    results.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': category_info.get('name', f'class_{cls_id}'),
                        'abbreviation': category_info.get('abbreviation', '')
                    })
        
        return results
    
    @torch.no_grad()
    def predict(
        self, 
        image: Union[str, np.ndarray, Image.Image]
    ) -> List[Dict]:
        """Run inference on a single image."""
        img_tensor, original_shape, preprocess_info = self.preprocess_image(image)
        
        outputs = self.model(img_tensor, self.text_embeddings)
        
        predictions = self.model.get_predictions(outputs)
        
        results = self.postprocess_predictions(
            predictions, original_shape, preprocess_info
        )
        
        return results
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8
    ) -> List[List[Dict]]:
        """Run inference on multiple images."""
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_results = [self.predict(img) for img in batch_images]
            all_results.extend(batch_results)
        
        return all_results
    
    def predict_with_timing(
        self, 
        image: Union[str, np.ndarray]
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """Run inference with timing information."""
        timings = {}
        
        start = time.time()
        img_tensor, original_shape, preprocess_info = self.preprocess_image(image)
        timings['preprocess'] = time.time() - start
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model(img_tensor, self.text_embeddings)
            predictions = self.model.get_predictions(outputs)
        timings['inference'] = time.time() - start
        
        start = time.time()
        results = self.postprocess_predictions(
            predictions, original_shape, preprocess_info
        )
        timings['postprocess'] = time.time() - start
        
        timings['total'] = sum(timings.values())
        timings['fps'] = 1.0 / timings['total'] if timings['total'] > 0 else 0
        
        return results, timings


def visualize_results(
    image_path: str,
    results: List[Dict],
    output_path: str,
    show_confidence: bool = True,
    line_thickness: int = 2
):
    """Visualize detection results on image."""
    img = cv2.imread(image_path)
    
    for det in results:
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        conf = det['confidence']
        cls_name = det.get('abbreviation') or det['class_name']
        
        color = get_category_color(det['class_id'])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        
        if show_confidence:
            label = f"{cls_name}: {conf:.2f}"
        else:
            label = cls_name
        
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img, (x1, y1 - text_h - baseline - 5), 
            (x1 + text_w, y1), color, -1
        )
        cv2.putText(
            img, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to {output_path}")


def get_category_color(class_id: int) -> Tuple[int, int, int]:
    """Generate consistent color for each category."""
    np.random.seed(class_id * 42)
    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
    return color


def main():
    parser = argparse.ArgumentParser(description='PLVLDet Inference')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Directory of images for batch inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--categories', type=str, required=True, help='Path to category file')
    parser.add_argument('--output_dir', type=str, default='results/inference',
                        help='Output directory for results')
    parser.add_argument('--conf_thresh', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization results')
    parser.add_argument('--save_json', action='store_true',
                        help='Save detection results as JSON')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark with timing information')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictor = PLVLDetPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        category_file=args.categories,
        device=args.device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh
    )
    
    if args.image:
        print(f"\nProcessing: {args.image}")
        
        if args.benchmark:
            results, timings = predictor.predict_with_timing(args.image)
            print(f"\nTiming breakdown:")
            print(f"  Preprocess: {timings['preprocess']*1000:.2f} ms")
            print(f"  Inference: {timings['inference']*1000:.2f} ms")
            print(f"  Postprocess: {timings['postprocess']*1000:.2f} ms")
            print(f"  Total: {timings['total']*1000:.2f} ms")
            print(f"  FPS: {timings['fps']:.1f}")
        else:
            results = predictor.predict(args.image)
        
        print(f"\nDetected {len(results)} objects:")
        for det in results:
            bbox = [f"{c:.1f}" for c in det['bbox']]
            print(f"  {det['class_name']} ({det['abbreviation']}): "
                  f"conf={det['confidence']:.3f}, bbox=[{', '.join(bbox)}]")
        
        if args.visualize:
            img_name = Path(args.image).stem
            vis_path = output_dir / f"{img_name}_vis.jpg"
            visualize_results(args.image, results, str(vis_path))
        
        if args.save_json:
            img_name = Path(args.image).stem
            json_path = output_dir / f"{img_name}_results.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'image': args.image,
                    'detections': results,
                    'num_detections': len(results)
                }, f, indent=2)
            print(f"Results saved to {json_path}")
    
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"\nFound {len(image_files)} images in {image_dir}")
        
        all_results = {}
        total_time = 0
        
        for img_path in tqdm(image_files, desc='Processing'):
            if args.benchmark:
                results, timings = predictor.predict_with_timing(str(img_path))
                total_time += timings['total']
            else:
                start = time.time()
                results = predictor.predict(str(img_path))
                total_time += time.time() - start
            
            all_results[img_path.name] = {
                'detections': results,
                'num_detections': len(results)
            }
            
            if args.visualize:
                vis_path = output_dir / f"{img_path.stem}_vis.jpg"
                visualize_results(str(img_path), results, str(vis_path))
        
        avg_time = total_time / len(image_files)
        print(f"\nBatch inference completed:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Average time per image: {avg_time*1000:.2f} ms")
        print(f"  Average FPS: {1/avg_time:.1f}")
        
        if args.save_json:
            json_path = output_dir / "batch_results.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'image_dir': str(image_dir),
                    'num_images': len(image_files),
                    'avg_time_ms': avg_time * 1000,
                    'results': all_results
                }, f, indent=2)
            print(f"Batch results saved to {json_path}")
    
    else:
        print("Error: Please specify --image or --image_dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
