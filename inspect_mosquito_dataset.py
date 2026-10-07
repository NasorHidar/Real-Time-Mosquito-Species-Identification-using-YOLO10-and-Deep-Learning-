#!/usr/bin/env python3
"""
YOLO Dataset Inspection Script for Mosquito Detection
=======================================================
Analyzes: total images, labels, class distribution, missing labels, malformed annotations
"""

import os
import sys
from pathlib import Path
from collections import Counter
import yaml

# --- CONFIGURATION ---
DATASET_PATH = r"F:\Research Project\Mosquito Detection Dataset.v4i.yolov11"
NUM_CLASSES = 6
# -------------------

def get_class_names(dataset_path):
    """Load class names from data.yaml"""
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'names' in config:
                    return config['names']
        except Exception as e:
            print(f"Error loading data.yaml: {e}")
    return [f"class_{i}" for i in range(NUM_CLASSES)]

def inspect_yolo_dataset(dataset_path: str):
    """Inspect a YOLO dataset and provide comprehensive statistics."""
    dataset_path = Path(dataset_path)
    class_names = get_class_names(dataset_path)
    
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'images_without_labels': [],
        'class_distribution': Counter(),
        'malformed_annotations': [],
        'splits': {}
    }
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
            
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Warning: Missing images or labels for {split}")
            continue
        
        # Get image and label files
        image_files = {f.stem: f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions}
        label_files = {f.stem: f for f in labels_dir.iterdir() if f.suffix == '.txt'}
        
        split_stats = {
            'images': len(image_files),
            'labels': len(label_files),
            'missing_labels': [],
            'class_counts': Counter(),
            'malformed': []
        }
        
        # Check each image for corresponding label
        for stem, img_path in image_files.items():
            stats['total_images'] += 1
            split_stats['images'] += 1
            
            if stem not in label_files:
                stats['images_without_labels'].append(str(img_path))
                split_stats['missing_labels'].append(stem)
            else:
                stats['total_labels'] += 1
                # Parse label file
                label_path = label_files[stem]
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) != 5:
                                stats['malformed_annotations'].append(
                                    f"{split}/{label_path.name}:{line_num} - Expected 5 values, got {len(parts)}"
                                )
                                split_stats['malformed'].append(stem)
                                continue
                            try:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Validate ranges
                                if not (0 <= class_id < NUM_CLASSES):
                                    stats['malformed_annotations'].append(
                                        f"{split}/{label_path.name}:{line_num} - Invalid class_id: {class_id}"
                                    )
                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                                    stats['malformed_annotations'].append(
                                        f"{split}/{label_path.name}:{line_num} - Coords out of [0,1]: {x_center}, {y_center}"
                                    )
                                if not (0 < width <= 1 and 0 < height <= 1):
                                    stats['malformed_annotations'].append(
                                        f"{split}/{label_path.name}:{line_num} - Dimensions out of (0,1]: {width}, {height}"
                                    )
                                
                                stats['class_distribution'][class_id] += 1
                                split_stats['class_counts'][class_id] += 1
                            except ValueError as e:
                                stats['malformed_annotations'].append(
                                    f"{split}/{label_path.name}:{line_num} - Parse error: {e}"
                                )
                                split_stats['malformed'].append(stem)
                except Exception as e:
                    stats['malformed_annotations'].append(f"{split}/{label_path.name}: Error reading - {e}")
        
        stats['splits'][split] = split_stats
        
        # Print split summary
        print(f"\n{'='*50}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'='*50}")
        print(f"  Images: {split_stats['images']}")
        print(f"  Labels: {split_stats['labels']}")
        print(f"  Missing labels: {len(split_stats['missing_labels'])}")
        print(f"  Malformed files: {len(split_stats['malformed'])}")
        
        if split_stats['class_counts']:
            print(f"  Class distribution:")
            for class_id, count in sorted(split_stats['class_counts'].items()):
                name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                print(f"    {class_id}: {name} -> {count} instances")
    
    # Print overall summary
    print(f"\n{'='*50}")
    print("OVERALL DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total images: {stats['total_images']}")
    print(f"Total label files: {stats['total_labels']}")
    print(f"Images without labels: {len(stats['images_without_labels'])}")
    print(f"Malformed annotations: {len(stats['malformed_annotations'])}")
    
    if stats['class_distribution']:
        print(f"\nTotal class distribution:")
        for class_id, count in sorted(stats['class_distribution'].items()):
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"  {class_id}: {name} -> {count} instances")
    
    # Warnings
    if stats['images_without_labels']:
        print(f"\nWARNING: {len(stats['images_without_labels'])} images without annotations!")
    if stats['malformed_annotations']:
        print(f"\nWARNING: {len(stats['malformed_annotations'])} malformed annotations found!")
    
    return stats

if __name__ == "__main__":
    print(f"Inspecting dataset: {DATASET_PATH}")
    inspect_yolo_dataset(DATASET_PATH)