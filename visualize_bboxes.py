#!/usr/bin/env python3
"""
Visualize random images with YOLO bounding boxes
=================================================
"""

import cv2
import random
from pathlib import Path
import yaml

# --- CONFIGURATION ---
DATASET_PATH = r"F:\Research Project\Mosquito Detection Dataset.v4i.yolov11"
NUM_SAMPLES = 5
SPLIT = 'train'  # train, valid, or test
# --------------------

def get_class_names(dataset_path):
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            if config and 'names' in config:
                return config['names']
    return [f"class_{i}" for i in range(6)]

def visualize_random_samples(dataset_path: str, num_samples: int = 5, split: str = 'train'):
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / split / 'images'
    labels_dir = dataset_path / split / 'labels'
    
    class_names = get_class_names(dataset_path)
    # Generate distinct colors for each class
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    random.shuffle(image_files)
    
    for img_path in image_files[:num_samples]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    box_w = float(parts[3]) * w
                    box_h = float(parts[4]) * h
                    
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    color = colors[class_id % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('YOLO Bounding Boxes - Press any key for next, ESC to exit', img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Visualizing {NUM_SAMPLES} samples from {SPLIT} split...")
    visualize_random_samples(DATASET_PATH, NUM_SAMPLES, SPLIT)