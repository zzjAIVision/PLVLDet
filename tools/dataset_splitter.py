#!/usr/bin/env python3
"""
Dataset Splitter for PLVLDet

Splits the dataset into training and evaluation sets based on transmission
line routes. Implements the paper's splitting strategy:
- MixPLOD: 142 transmission lines for pretraining (21 seen categories)
- PLEval: 47 transmission lines for evaluation (32 categories)

Usage:
    python tools/dataset_splitter.py \
        --patch_dir data/patches \
        --output_dir data/splits \
        --pretrain_ratio 0.75 \
        --val_ratio 0.1 \
        --test_ratio 0.15
"""

import os
import sys
import json
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# Seen categories for pretraining (leaf categories from seen parent nodes)
# Based on paper: 14 seen leaf categories + 7 parent categories = 21
SEEN_CATEGORIES = {
    # Parent categories (always seen)
    0,   # IT - Insulator
    1,   # CI - Ceramic insulator
    2,   # GI - Glass insulator
    3,   # PI - Polymer insulator
    15,  # FMT - Foreign matters on tower
    20,  # WFM - Wire foreign matter
    26,  # VD - vibration damper
    # Leaf categories (seen during pretraining)
    4,   # NGI - Normal glass insulator
    5,   # BGI - Broken glass insulator
    11,  # NCI - Normal ceramic insulator
    12,  # CCI - Contaminated ceramic insulator
    16,  # BNT - Bird's nest on tower
    17,  # PBT - Plastic bag on tower
    21,  # WPB - Wire plastic bag
    24,  # BSTL - Broken strand of transmission line
    25,  # LSTL - Loose strand of transmission line
    27,  # DD - Displacement of damper
    29,  # TC - Tension clamp
    30,  # AC - armour clamp
}

# Unseen categories for zero-shot evaluation
UNSEEN_CATEGORIES = {
    6,   # CGI - Contaminated glass insulator
    7,   # NPI - Normal Polymer insulator
    8,   # DaPI - Damaged Polymer insulator
    9,   # CPI - Contaminated Polymer insulator
    10,  # DePI - Deformed Polymer insulator
    13,  # BCI - Broken ceramic insulator
    14,  # FoCI - Fall-off ceramic insulator
    18,  # KoT - Kite on tower
    19,  # TB - Tower balloon
    22,  # WK - Wire kite
    23,  # WB - Wire balloon
    28,  # MD - Missing damper
    31,  # MSP - Missing split pin
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


class DatasetSplitter:
    """Splits dataset into train/val/test at route level."""
    
    def __init__(
        self,
        patch_dir,
        output_dir,
        pretrain_ratio=0.75,
        val_ratio=0.1,
        test_ratio=0.15,
        seed=42
    ):
        """
        Initialize dataset splitter.
        
        Args:
            patch_dir: Directory containing patches
            output_dir: Output directory for split files
            pretrain_ratio: Ratio of routes for pretraining (MixPLOD)
            val_ratio: Ratio of PLEval routes for validation
            test_ratio: Ratio of PLEval routes for testing
            seed: Random seed for reproducibility
        """
        self.patch_dir = Path(patch_dir)
        self.output_dir = Path(output_dir)
        self.pretrain_ratio = pretrain_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        random.seed(seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch metadata
        self.load_metadata()
        
    def load_metadata(self):
        """Load patch metadata and organize by route."""
        metadata_path = self.patch_dir / 'patch_metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.patches = self.metadata.get('patches', [])
        else:
            # Scan patch directory
            self.patches = []
            label_dir = self.patch_dir / 'labels'
            
            for label_file in label_dir.glob('*.txt'):
                patch_name = label_file.stem
                
                # Parse categories from label file
                categories = []
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            categories.append(int(parts[0]))
                            
                # Extract source image name (route identifier)
                # Assuming format: {route_id}_{image_id}_patch_{patch_id}
                source_parts = patch_name.rsplit('_patch_', 1)
                source_image = source_parts[0] if len(source_parts) > 1 else patch_name
                
                self.patches.append({
                    'patch_name': patch_name,
                    'source_image': source_image,
                    'categories': categories
                })
                
        # Extract unique routes
        self.routes = defaultdict(list)
        for patch in self.patches:
            # Extract route identifier from source image
            source = patch['source_image']
            # Assume route is first part before underscore or first segment
            route_id = self._extract_route_id(source)
            self.routes[route_id].append(patch)
            
        print(f"Loaded {len(self.patches)} patches from {len(self.routes)} routes")
        
    def _extract_route_id(self, source_image):
        """Extract route identifier from source image name."""
        # Different naming conventions
        # Try to extract route/transmission line identifier
        name = Path(source_image).stem
        
        # Common patterns:
        # - "route001_img001" -> "route001"
        # - "line_001_001" -> "line_001"
        # - "TL001_20230101_001" -> "TL001"
        
        parts = name.split('_')
        if len(parts) >= 2:
            # Check if first part looks like a route ID
            if any(x in parts[0].lower() for x in ['route', 'line', 'tl', 'pl']):
                return parts[0]
            # Otherwise use first two parts
            return '_'.join(parts[:2])
        return name
        
    def analyze_route_categories(self):
        """Analyze category distribution per route."""
        route_stats = {}
        
        for route_id, patches in self.routes.items():
            categories = set()
            cat_counts = defaultdict(int)
            
            for patch in patches:
                for cat_id in patch['categories']:
                    categories.add(cat_id)
                    cat_counts[cat_id] += 1
                    
            has_unseen = bool(categories & UNSEEN_CATEGORIES)
            has_seen_only = categories.issubset(SEEN_CATEGORIES)
            
            route_stats[route_id] = {
                'num_patches': len(patches),
                'categories': list(categories),
                'category_counts': dict(cat_counts),
                'has_unseen': has_unseen,
                'has_seen_only': has_seen_only
            }
            
        return route_stats
        
    def split_routes(self):
        """
        Split routes into MixPLOD (pretrain) and PLEval (finetune/eval).
        
        Returns routes containing unseen categories to PLEval.
        """
        route_stats = self.analyze_route_categories()
        
        # Separate routes by category presence
        routes_with_unseen = []
        routes_seen_only = []
        
        for route_id, stats in route_stats.items():
            if stats['has_unseen']:
                routes_with_unseen.append(route_id)
            else:
                routes_seen_only.append(route_id)
                
        print(f"\nRoutes with unseen categories: {len(routes_with_unseen)}")
        print(f"Routes with seen categories only: {len(routes_seen_only)}")
        
        # Shuffle for random split
        random.shuffle(routes_with_unseen)
        random.shuffle(routes_seen_only)
        
        # Allocate routes
        # MixPLOD: majority of seen-only routes
        # PLEval: routes with unseen + some seen routes for diversity
        
        total_routes = len(self.routes)
        pretrain_count = int(total_routes * self.pretrain_ratio)
        
        # MixPLOD gets most seen-only routes
        mixplod_routes = routes_seen_only[:min(len(routes_seen_only), pretrain_count)]
        
        # PLEval gets routes with unseen categories + remaining
        pleval_routes = routes_with_unseen + routes_seen_only[len(mixplod_routes):]
        
        # Further split PLEval into train/val/test
        random.shuffle(pleval_routes)
        
        pleval_total = len(pleval_routes)
        val_count = int(pleval_total * self.val_ratio / (self.val_ratio + self.test_ratio + (1 - self.pretrain_ratio)))
        test_count = int(pleval_total * self.test_ratio / (self.val_ratio + self.test_ratio + (1 - self.pretrain_ratio)))
        train_count = pleval_total - val_count - test_count
        
        pleval_train = pleval_routes[:train_count]
        pleval_val = pleval_routes[train_count:train_count + val_count]
        pleval_test = pleval_routes[train_count + val_count:]
        
        splits = {
            'mixplod_pretrain': mixplod_routes,
            'pleval_train': pleval_train,
            'pleval_val': pleval_val,
            'pleval_test': pleval_test
        }
        
        return splits
        
    def create_split_files(self, splits):
        """Create split files with patch names."""
        split_info = {}
        
        for split_name, route_ids in splits.items():
            patches_in_split = []
            categories_in_split = set()
            
            for route_id in route_ids:
                for patch in self.routes[route_id]:
                    patches_in_split.append(patch['patch_name'])
                    categories_in_split.update(patch['categories'])
                    
            # Write patch list
            split_file = self.output_dir / f'{split_name}.txt'
            with open(split_file, 'w') as f:
                for patch_name in sorted(patches_in_split):
                    f.write(f"{patch_name}\n")
                    
            split_info[split_name] = {
                'num_routes': len(route_ids),
                'num_patches': len(patches_in_split),
                'categories': sorted(list(categories_in_split)),
                'routes': route_ids
            }
            
            print(f"\n{split_name}:")
            print(f"  Routes: {len(route_ids)}")
            print(f"  Patches: {len(patches_in_split)}")
            print(f"  Categories: {len(categories_in_split)}")
            
        # Save split metadata
        metadata_file = self.output_dir / 'split_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(split_info, f, indent=2)
            
        return split_info
        
    def create_symlinks(self, splits):
        """Create symlinked directories for each split."""
        image_dir = self.patch_dir / 'images'
        label_dir = self.patch_dir / 'labels'
        
        for split_name, route_ids in splits.items():
            split_image_dir = self.output_dir / split_name / 'images'
            split_label_dir = self.output_dir / split_name / 'labels'
            
            split_image_dir.mkdir(parents=True, exist_ok=True)
            split_label_dir.mkdir(parents=True, exist_ok=True)
            
            for route_id in route_ids:
                for patch in self.routes[route_id]:
                    patch_name = patch['patch_name']
                    
                    # Image symlink
                    src_image = image_dir / f"{patch_name}.jpg"
                    dst_image = split_image_dir / f"{patch_name}.jpg"
                    
                    if src_image.exists() and not dst_image.exists():
                        try:
                            dst_image.symlink_to(src_image.resolve())
                        except OSError:
                            shutil.copy2(src_image, dst_image)
                            
                    # Label symlink
                    src_label = label_dir / f"{patch_name}.txt"
                    dst_label = split_label_dir / f"{patch_name}.txt"
                    
                    if src_label.exists() and not dst_label.exists():
                        try:
                            dst_label.symlink_to(src_label.resolve())
                        except OSError:
                            shutil.copy2(src_label, dst_label)
                            
    def generate_category_stats(self, splits):
        """Generate detailed category statistics for each split."""
        stats_file = self.output_dir / 'category_statistics.txt'
        
        with open(stats_file, 'w') as f:
            f.write("PLVLDet Dataset Category Statistics\n")
            f.write("=" * 60 + "\n\n")
            
            for split_name, route_ids in splits.items():
                f.write(f"\n{split_name.upper()}\n")
                f.write("-" * 40 + "\n")
                
                cat_counts = defaultdict(int)
                
                for route_id in route_ids:
                    for patch in self.routes[route_id]:
                        for cat_id in patch['categories']:
                            cat_counts[cat_id] += 1
                            
                total = sum(cat_counts.values())
                f.write(f"Total annotations: {total}\n\n")
                
                # Seen categories
                f.write("Seen categories:\n")
                for cat_id in sorted(SEEN_CATEGORIES):
                    count = cat_counts.get(cat_id, 0)
                    name = CATEGORY_NAMES[cat_id]
                    f.write(f"  [{cat_id:2d}] {name:40s}: {count:6d}\n")
                    
                # Unseen categories
                f.write("\nUnseen categories:\n")
                for cat_id in sorted(UNSEEN_CATEGORIES):
                    count = cat_counts.get(cat_id, 0)
                    name = CATEGORY_NAMES[cat_id]
                    marker = " *" if count > 0 else ""
                    f.write(f"  [{cat_id:2d}] {name:40s}: {count:6d}{marker}\n")
                    
                f.write("\n")
                
        print(f"\nCategory statistics saved to: {stats_file}")
        
    def run(self):
        """Run the complete splitting pipeline."""
        print("Analyzing routes and categories...")
        
        splits = self.split_routes()
        
        print("\nCreating split files...")
        split_info = self.create_split_files(splits)
        
        print("\nCreating symlinked directories...")
        self.create_symlinks(splits)
        
        print("\nGenerating category statistics...")
        self.generate_category_stats(splits)
        
        # Summary
        print("\n" + "=" * 60)
        print("Dataset Splitting Complete")
        print("=" * 60)
        
        total_patches = sum(info['num_patches'] for info in split_info.values())
        print(f"\nTotal patches: {total_patches}")
        
        for split_name, info in split_info.items():
            pct = info['num_patches'] / total_patches * 100 if total_patches > 0 else 0
            print(f"  {split_name}: {info['num_patches']} patches ({pct:.1f}%)")
            
        print(f"\nOutput directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test sets at route level'
    )
    parser.add_argument(
        '--patch_dir', type=str, default='data/patches',
        help='Directory containing patches'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/splits',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--pretrain_ratio', type=float, default=0.75,
        help='Ratio of routes for pretraining (MixPLOD)'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help='Ratio of PLEval routes for validation'
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.15,
        help='Ratio of PLEval routes for testing'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        pretrain_ratio=args.pretrain_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    splitter.run()


if __name__ == '__main__':
    main()
