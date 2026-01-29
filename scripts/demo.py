#!/usr/bin/env python3
"""
PLVLDet Demo Script

Interactive demonstration of PLVLDet for power line component detection.
Supports:
- Single image demo with detailed visualization
- Video demo with real-time detection
- Comparison with baseline models
- Zero-shot detection demonstration
"""

import os
import sys
import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference import PLVLDetPredictor, get_category_color


CATEGORY_COLORS = {
    # Insulators - Blue shades
    'IT': (255, 100, 100), 'CI': (255, 120, 80), 'GI': (255, 140, 60),
    'PI': (255, 160, 40), 'NGI': (200, 255, 100), 'BGI': (100, 100, 255),
    'CGI': (150, 100, 255), 'NPI': (200, 255, 150), 'DaPI': (100, 150, 255),
    'CPI': (150, 150, 255), 'DePI': (180, 100, 255), 'NCI': (200, 255, 200),
    'CCI': (150, 200, 255), 'BCI': (100, 200, 255), 'FoCI': (80, 180, 255),
    # Foreign matters - Green shades
    'FMT': (100, 255, 100), 'BNT': (80, 255, 120), 'PBT': (60, 255, 140),
    'KoT': (40, 255, 160), 'TB': (20, 255, 180),
    # Wire foreign matters - Yellow shades
    'WFM': (100, 255, 255), 'WPB': (80, 255, 255), 'WK': (60, 255, 255),
    'WB': (40, 255, 255),
    # Transmission line - Orange shades
    'BSTL': (100, 180, 255), 'LSTL': (80, 160, 255),
    # Vibration damper - Purple shades
    'VD': (255, 100, 255), 'DD': (255, 80, 220), 'MD': (255, 60, 200),
    # Clamps - Cyan shades
    'TC': (255, 255, 100), 'AC': (255, 255, 80), 'MSP': (255, 255, 60),
}


class PLVLDetDemo:
    """Interactive demo for PLVLDet."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        category_file: str,
        device: str = 'cuda'
    ):
        self.predictor = PLVLDetPredictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            category_file=category_file,
            device=device
        )
        
        with open(category_file, 'r') as f:
            self.category_config = yaml.safe_load(f)
    
    def get_color(self, class_id: int, abbreviation: str) -> Tuple[int, int, int]:
        """Get color for category."""
        if abbreviation in CATEGORY_COLORS:
            return CATEGORY_COLORS[abbreviation]
        return get_category_color(class_id)
    
    def create_detection_visualization(
        self,
        image_path: str,
        results: List[Dict],
        output_path: str,
        title: str = "PLVLDet Detection Results",
        figsize: Tuple[int, int] = (16, 10)
    ):
        """Create detailed visualization with matplotlib."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(img)
        axes[1].set_title(f"{title}\n({len(results)} detections)", fontsize=12)
        axes[1].axis('off')
        
        legend_elements = []
        seen_classes = set()
        
        for det in results:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            abbr = det.get('abbreviation', '')
            cls_id = det['class_id']
            
            color = np.array(self.get_color(cls_id, abbr)) / 255.0
            
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[1].add_patch(rect)
            
            label = f"{abbr}: {conf:.2f}" if abbr else f"{cls_name[:15]}: {conf:.2f}"
            axes[1].text(
                x1, y1 - 5, label,
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
            )
            
            if cls_id not in seen_classes:
                seen_classes.add(cls_id)
                legend_elements.append(
                    plt.Line2D([0], [0], color=color, linewidth=3, 
                              label=f"{abbr}: {cls_name}" if abbr else cls_name)
                )
        
        if legend_elements:
            axes[1].legend(
                handles=legend_elements[:15],
                loc='upper right',
                fontsize=8,
                framealpha=0.9
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def create_category_distribution_chart(
        self,
        results: List[Dict],
        output_path: str
    ):
        """Create bar chart of detected categories."""
        category_counts = {}
        for det in results:
            cls_name = det.get('abbreviation') or det['class_name']
            category_counts[cls_name] = category_counts.get(cls_name, 0) + 1
        
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = [c[0] for c in sorted_cats]
        counts = [c[1] for c in sorted_cats]
        
        colors = [np.array(CATEGORY_COLORS.get(c, (100, 150, 200))) / 255.0 
                  for c in categories]
        
        bars = ax.barh(categories, counts, color=colors)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   str(count), va='center', fontsize=10)
        
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        ax.set_title('Detection Category Distribution', fontsize=14)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution chart saved to {output_path}")
    
    def run_image_demo(
        self,
        image_path: str,
        output_dir: str,
        conf_thresh: float = 0.25,
        show_distribution: bool = True
    ):
        """Run demo on single image with comprehensive output."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictor.conf_thresh = conf_thresh
        
        print(f"\n{'='*60}")
        print(f"PLVLDet Demo - Single Image")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Confidence threshold: {conf_thresh}")
        
        results, timings = self.predictor.predict_with_timing(image_path)
        
        print(f"\nInference completed:")
        print(f"  Total time: {timings['total']*1000:.2f} ms")
        print(f"  FPS: {timings['fps']:.1f}")
        print(f"  Detections: {len(results)}")
        
        img_name = Path(image_path).stem
        
        vis_path = output_dir / f"{img_name}_detection.png"
        self.create_detection_visualization(
            image_path, results, str(vis_path),
            title="PLVLDet Detection Results"
        )
        
        if show_distribution and len(results) > 0:
            dist_path = output_dir / f"{img_name}_distribution.png"
            self.create_category_distribution_chart(results, str(dist_path))
        
        json_path = output_dir / f"{img_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'image': image_path,
                'conf_threshold': conf_thresh,
                'num_detections': len(results),
                'inference_time_ms': timings['total'] * 1000,
                'detections': results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        
        return results
    
    def run_comparison_demo(
        self,
        image_path: str,
        output_dir: str,
        conf_thresholds: List[float] = [0.1, 0.25, 0.5, 0.75]
    ):
        """Run demo comparing different confidence thresholds."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"PLVLDet Demo - Confidence Threshold Comparison")
        print(f"{'='*60}")
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        n_thresh = len(conf_thresholds)
        fig, axes = plt.subplots(2, (n_thresh + 1) // 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, conf_thresh in enumerate(conf_thresholds):
            self.predictor.conf_thresh = conf_thresh
            results = self.predictor.predict(image_path)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"Conf={conf_thresh} ({len(results)} detections)", fontsize=11)
            axes[idx].axis('off')
            
            for det in results:
                x1, y1, x2, y2 = det['bbox']
                color = np.array(self.get_color(
                    det['class_id'], 
                    det.get('abbreviation', '')
                )) / 255.0
                
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor=color, facecolor='none'
                )
                axes[idx].add_patch(rect)
        
        for idx in range(n_thresh, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Effect of Confidence Threshold on Detection", fontsize=14)
        plt.tight_layout()
        
        output_path = output_dir / f"{Path(image_path).stem}_threshold_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison saved to {output_path}")
    
    def run_video_demo(
        self,
        video_path: str,
        output_path: str,
        conf_thresh: float = 0.25,
        max_frames: int = None,
        show_fps: bool = True
    ):
        """Run demo on video file."""
        print(f"\n{'='*60}")
        print(f"PLVLDet Demo - Video Processing")
        print(f"{'='*60}")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        
        self.predictor.conf_thresh = conf_thresh
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_times = []
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        frame_count = 0
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.predictor.predict(frame_rgb)
            
            inference_time = time.time() - start_time
            frame_times.append(inference_time)
            
            for det in results:
                x1, y1, x2, y2 = [int(c) for c in det['bbox']]
                conf = det['confidence']
                abbr = det.get('abbreviation', det['class_name'][:10])
                cls_id = det['class_id']
                
                color = self.get_color(cls_id, abbr)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{abbr}: {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(frame, (x1, y1 - text_h - 5), 
                            (x1 + text_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if show_fps:
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                fps_text = f"FPS: {current_fps:.1f} | Detections: {len(results)}"
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        avg_time = np.mean(frame_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\nVideo processing completed:")
        print(f"  Output: {output_path}")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average inference time: {avg_time*1000:.2f} ms")
        print(f"  Average FPS: {avg_fps:.1f}")
    
    def run_zeroshot_demo(
        self,
        image_path: str,
        custom_categories: List[str],
        output_dir: str,
        conf_thresh: float = 0.1
    ):
        """Demonstrate zero-shot detection with custom category prompts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"PLVLDet Demo - Zero-Shot Detection")
        print(f"{'='*60}")
        print(f"Custom categories: {custom_categories}")
        
        original_embeddings = self.predictor.text_embeddings
        original_categories = self.predictor.categories
        
        custom_embeddings = self.predictor.model.encode_text(custom_categories)
        custom_embeddings = custom_embeddings.to(self.predictor.device)
        
        self.predictor.text_embeddings = custom_embeddings
        self.predictor.categories = {
            i: {'name': cat, 'abbreviation': '', 'prompt': cat}
            for i, cat in enumerate(custom_categories)
        }
        self.predictor.num_classes = len(custom_categories)
        self.predictor.conf_thresh = conf_thresh
        
        results = self.predictor.predict(image_path)
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img)
        ax.set_title(f"Zero-Shot Detection\nCategories: {', '.join(custom_categories)}", 
                    fontsize=12)
        ax.axis('off')
        
        for det in results:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            
            color = get_category_color(det['class_id'])
            color = np.array(color) / 255.0
            
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(
                x1, y1 - 5, f"{cls_name}: {conf:.2f}",
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
            )
        
        plt.tight_layout()
        output_path = output_dir / f"{Path(image_path).stem}_zeroshot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.predictor.text_embeddings = original_embeddings
        self.predictor.categories = original_categories
        self.predictor.num_classes = len(original_categories)
        
        print(f"Zero-shot results saved to {output_path}")
        print(f"Detected {len(results)} objects")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='PLVLDet Interactive Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--categories', type=str, required=True,
                        help='Path to category file')
    parser.add_argument('--output_dir', type=str, default='results/demo',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    subparsers = parser.add_subparsers(dest='mode', help='Demo mode')
    
    image_parser = subparsers.add_parser('image', help='Single image demo')
    image_parser.add_argument('--image', type=str, required=True,
                             help='Input image path')
    image_parser.add_argument('--conf_thresh', type=float, default=0.25,
                             help='Confidence threshold')
    
    compare_parser = subparsers.add_parser('compare', 
                                           help='Threshold comparison demo')
    compare_parser.add_argument('--image', type=str, required=True,
                               help='Input image path')
    compare_parser.add_argument('--thresholds', type=float, nargs='+',
                               default=[0.1, 0.25, 0.5, 0.75],
                               help='Confidence thresholds to compare')
    
    video_parser = subparsers.add_parser('video', help='Video demo')
    video_parser.add_argument('--video', type=str, required=True,
                             help='Input video path')
    video_parser.add_argument('--output', type=str, default=None,
                             help='Output video path')
    video_parser.add_argument('--conf_thresh', type=float, default=0.25,
                             help='Confidence threshold')
    video_parser.add_argument('--max_frames', type=int, default=None,
                             help='Maximum frames to process')
    
    zeroshot_parser = subparsers.add_parser('zeroshot', 
                                            help='Zero-shot detection demo')
    zeroshot_parser.add_argument('--image', type=str, required=True,
                                help='Input image path')
    zeroshot_parser.add_argument('--custom_categories', type=str, nargs='+',
                                required=True,
                                help='Custom category prompts for zero-shot')
    zeroshot_parser.add_argument('--conf_thresh', type=float, default=0.1,
                                help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    demo = PLVLDetDemo(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        category_file=args.categories,
        device=args.device
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'image':
        demo.run_image_demo(
            args.image,
            str(output_dir),
            conf_thresh=args.conf_thresh
        )
    
    elif args.mode == 'compare':
        demo.run_comparison_demo(
            args.image,
            str(output_dir),
            conf_thresholds=args.thresholds
        )
    
    elif args.mode == 'video':
        output_video = args.output or str(
            output_dir / f"{Path(args.video).stem}_output.mp4"
        )
        demo.run_video_demo(
            args.video,
            output_video,
            conf_thresh=args.conf_thresh,
            max_frames=args.max_frames
        )
    
    elif args.mode == 'zeroshot':
        demo.run_zeroshot_demo(
            args.image,
            args.custom_categories,
            str(output_dir),
            conf_thresh=args.conf_thresh
        )


if __name__ == '__main__':
    main()
