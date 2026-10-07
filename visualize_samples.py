"""
Visualize Dataset Samples
=========================
Stage 3: Visualize random samples from the prepared dataset

This script:
1. Loads random images from each split
2. Draws bounding boxes with class labels
3. Displays them in a grid for visual verification
"""

import cv2
import numpy as np
import random
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"E:\Claude local session\mosquito-detection")

# Number of samples to visualize per split
SAMPLES_PER_SPLIT = 3

# Colors for each class (BGR for OpenCV)
COLORS = [
    (255, 0, 0),    # Blue - aegypti
    (0, 255, 0),    # Green - albopictus
    (0, 0, 255),    # Red - anopheles
    (255, 255, 0),  # Cyan - culex
    (255, 0, 255),  # Magenta - culiseta
    (0, 255, 255),  # Yellow - japonicus-koreicus
]

# =============================================================================


def load_config():
    """Load data.yaml configuration"""
    yaml_path = PROJECT_ROOT / "data.yaml"
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def draw_boxes_on_image(img_path, label_path, class_names):
    """
    Draw bounding boxes on an image.

    Returns the image with boxes drawn.
    """
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Could not load image: {img_path}")
        return None

    img_h, img_w = img.shape[:2]

    # Load annotations
    if not label_path.exists():
        print(f"[WARNING] No label file for: {img_path.name}")
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Draw each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h

            # Calculate corners
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Get color and class name
            color = COLORS[class_id % len(COLORS)]
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_text = f"{class_name}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)

            # Draw label text
            cv2.putText(img, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw center point
            cv2.circle(img, (int(x_center), int(y_center)), 3, color, -1)

        except Exception as e:
            print(f"[ERROR] Failed to draw box: {e}")
            continue

    return img


def visualize_samples_opencv(split='train', num_samples=5):
    """
    Visualize samples using OpenCV (one at a time).
    """
    config = load_config()
    class_names = config['names']

    images_dir = PROJECT_ROOT / "dataset" / "images" / split
    labels_dir = PROJECT_ROOT / "dataset" / "labels" / split

    # Get random images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return

    random.shuffle(image_files)
    samples = image_files[:num_samples]

    print(f"\nVisualizing {len(samples)} samples from {split} split...")
    print("Press any key to see next image, ESC to exit")

    for img_path in samples:
        label_path = labels_dir / f"{img_path.stem}.txt"

        img_with_boxes = draw_boxes_on_image(img_path, label_path, class_names)

        if img_with_boxes is not None:
            # Add title
            title = f"{split.upper()}: {img_path.name[:40]}..."
            cv2.putText(img_with_boxes, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Dataset Samples', img_with_boxes)
            key = cv2.waitKey(0)

            if key == 27:  # ESC
                break

    cv2.destroyAllWindows()


def visualize_samples_matplotlib(num_samples_per_split=3):
    """
    Visualize samples from all splits in a grid using matplotlib.
    """
    config = load_config()
    class_names = config['names']

    splits = ['train', 'val', 'test']

    # Collect samples
    all_samples = []

    for split in splits:
        images_dir = PROJECT_ROOT / "dataset" / "images" / split
        labels_dir = PROJECT_ROOT / "dataset" / "labels" / split

        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        if not image_files:
            continue

        random.shuffle(image_files)
        samples = image_files[:num_samples_per_split]

        for img_path in samples:
            label_path = labels_dir / f"{img_path.stem}.txt"
            img = draw_boxes_on_image(img_path, label_path, class_names)
            if img is not None:
                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_samples.append((split, img_path.name, img_rgb))

    if not all_samples:
        print("[ERROR] No samples to display")
        return

    # Create grid
    num_samples = len(all_samples)
    cols = 3
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (split, filename, img) in enumerate(all_samples):
        row = idx // cols
        col = idx % cols

        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(f"{split.upper()}: {filename[:30]}...", fontsize=10)
        ax.axis('off')

    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.suptitle('Dataset Samples with Bounding Boxes', fontsize=14, y=1.002)
    plt.show()


def print_sample_statistics():
    """Print statistics about random samples"""
    config = load_config()
    class_names = config['names']

    print("\n" + "="*60)
    print("SAMPLE STATISTICS")
    print("="*60)

    for split in ['train', 'val', 'test']:
        labels_dir = PROJECT_ROOT / "dataset" / "labels" / split
        label_files = list(labels_dir.glob('*.txt'))

        if not label_files:
            continue

        # Sample 100 random files or all if less than 100
        sample_size = min(100, len(label_files))
        sample_files = random.sample(label_files, sample_size)

        total_objects = 0
        class_counts = {i: 0 for i in range(len(class_names))}
        bbox_sizes = []

        for label_path in sample_files:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])

                    total_objects += 1
                    class_counts[class_id] += 1
                    bbox_sizes.append((width, height))
                except:
                    continue

        print(f"\n{split.upper()} (sampled {sample_size} files):")
        print(f"  Total objects: {total_objects}")
        print(f"  Avg objects per image: {total_objects / sample_size:.1f}")

        if bbox_sizes:
            widths = [w for w, h in bbox_sizes]
            heights = [h for w, h in bbox_sizes]
            print(f"  Bbox width:  min={min(widths):.3f}, max={max(widths):.3f}, avg={np.mean(widths):.3f}")
            print(f"  Bbox height: min={min(heights):.3f}, max={max(heights):.3f}, avg={np.mean(heights):.3f}")

        print(f"  Class distribution:")
        for class_id, count in class_counts.items():
            if count > 0:
                percentage = (count / total_objects * 100) if total_objects > 0 else 0
                print(f"    {class_names[class_id]:20s}: {count:4d} ({percentage:5.1f}%)")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("DATASET SAMPLE VISUALIZATION")
    print("="*60)
    print(f"Project: {PROJECT_ROOT}")

    if not PROJECT_ROOT.exists():
        print(f"\n[ERROR] Project not found: {PROJECT_ROOT}")
        print("Please run setup_project_structure.py first")
        return

    # Print sample statistics
    print_sample_statistics()

    # Visualize using matplotlib (grid view)
    print("\n" + "="*60)
    print("Opening matplotlib visualization...")
    print("Close the window to continue")
    print("="*60)
    visualize_samples_matplotlib(num_samples_per_split=SAMPLES_PER_SPLIT)

    # Optional: OpenCV one-by-one view
    print("\nWould you like to see individual samples? (OpenCV)")
    response = input("Enter 'y' to continue, or any other key to exit: ")

    if response.lower() == 'y':
        for split in ['train', 'val', 'test']:
            print(f"\nShowing {split} samples...")
            visualize_samples_opencv(split=split, num_samples=5)


if __name__ == "__main__":
    main()
