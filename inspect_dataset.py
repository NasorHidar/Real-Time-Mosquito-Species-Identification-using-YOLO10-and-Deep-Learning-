"""
Comprehensive Dataset Inspector for Object Detection
======================================================
Detects and analyzes: YOLO, COCO JSON, and Pascal VOC XML formats

This script will:
1. Auto-detect annotation format
2. Count images and labels
3. Analyze class distribution
4. Find missing labels
5. Detect malformed annotations
6. Generate statistics and visualizations
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import yaml
import random

# Set matplotlib backend for Windows
import matplotlib
matplotlib.use('TkAgg')

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_header(text):
    """Print a colored header"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}[OK] {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}[ERROR] {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}[WARNING] {text}{RESET}")

def print_info(text):
    """Print info message"""
    print(f"{CYAN}[INFO] {text}{RESET}")


class DatasetInspector:
    """
    Inspects object detection datasets in YOLO, COCO, or Pascal VOC format.
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.format = None
        self.classes = []
        self.num_classes = 0
        self.splits = {}
        self.stats = {
            'images_per_split': {},
            'labels_per_split': {},
            'class_distribution': Counter(),
            'missing_labels': [],
            'empty_labels': [],
            'malformed_annotations': [],
            'bbox_stats': {'widths': [], 'heights': [], 'areas': []}
        }

    def detect_format(self):
        """
        Auto-detect the annotation format (YOLO, COCO, or Pascal VOC)
        """
        print_header("DETECTING ANNOTATION FORMAT")

        # Check for YOLO format (data.yaml + .txt files)
        yaml_path = self.dataset_path / 'data.yaml'
        txt_files = list(self.dataset_path.rglob('*.txt'))

        # Check for COCO format (.json annotation files)
        json_files = list(self.dataset_path.rglob('*.json'))

        # Check for Pascal VOC format (.xml annotation files)
        xml_files = list(self.dataset_path.rglob('*.xml'))

        # Determine format based on what exists
        formats_found = []

        if yaml_path.exists() and txt_files:
            formats_found.append('YOLO')

        if json_files:
            # Check if it's COCO format JSON
            for jf in json_files[:3]:  # Check first 3 JSON files
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                    if 'images' in data and 'annotations' in data:
                        formats_found.append('COCO')
                        break
                except:
                    pass

        if xml_files:
            # Check if it's Pascal VOC format XML
            for xf in xml_files[:3]:
                try:
                    tree = ET.parse(xf)
                    root = tree.getroot()
                    if root.find('object') is not None:
                        formats_found.append('Pascal_VOC')
                        break
                except:
                    pass

        # Report findings
        print_info(f"Found: {len(txt_files)} .txt files")
        print_info(f"Found: {len(json_files)} .json files")
        print_info(f"Found: {len(xml_files)} .xml files")

        if yaml_path.exists():
            print_info(f"Found: data.yaml")

        # Determine primary format
        if 'YOLO' in formats_found:
            self.format = 'YOLO'
            print_success(f"Detected format: YOLO (txt + data.yaml)")
        elif 'COCO' in formats_found:
            self.format = 'COCO'
            print_success(f"Detected format: COCO JSON")
        elif 'Pascal_VOC' in formats_found:
            self.format = 'Pascal_VOC'
            print_success(f"Detected format: Pascal VOC XML")
        else:
            print_error("Could not detect annotation format!")
            print_info("Supported formats: YOLO (txt), COCO (JSON), Pascal VOC (XML)")

        return self.format

    def inspect_yolo(self):
        """
        Inspect YOLO format dataset
        """
        print_header("INSPECTING YOLO FORMAT")

        # Read data.yaml
        yaml_path = self.dataset_path / 'data.yaml'

        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            self.num_classes = data.get('nc', 0)
            self.classes = data.get('names', [])

            print_success(f"data.yaml found")
            print_info(f"Number of classes: {self.num_classes}")
            print_info(f"Class names: {self.classes}")

            # Detect splits
            for split in ['train', 'valid', 'val', 'test']:
                if split in data:
                    split_name = 'valid' if split == 'val' else split
                    split_path = self.dataset_path / split
                    if split_path.exists():
                        self.splits[split_name] = split_path
        else:
            print_warning("No data.yaml found, searching for folders...")
            # Try to detect splits from folder structure
            for split in ['train', 'valid', 'val', 'test']:
                split_path = self.dataset_path / split
                if split_path.exists():
                    split_name = 'valid' if split == 'val' else split
                    self.splits[split_name] = split_path

        # Analyze each split
        for split_name, split_path in self.splits.items():
            self._analyze_yolo_split(split_name, split_path)

    def _analyze_yolo_split(self, split_name, split_path):
        """
        Analyze a single YOLO split (train/valid/test)
        """
        print(f"\n{CYAN}Analyzing {split_name.upper()} split:{RESET}")

        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'

        # If no images/ subdirectory, check if images are in split root
        if not images_dir.exists():
            images_dir = split_path

        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [f for f in images_dir.rglob('*') if f.suffix.lower() in image_extensions]

        # Find all labels
        labels = list(labels_dir.rglob('*.txt')) if labels_dir.exists() else []

        self.stats['images_per_split'][split_name] = len(images)
        self.stats['labels_per_split'][split_name] = len(labels)

        print_info(f"Images: {len(images)}")
        print_info(f"Labels: {len(labels)}")

        # Check for matching pairs
        image_stems = {f.stem for f in images}
        label_stems = {f.stem for f in labels}

        missing_labels = image_stems - label_stems
        extra_labels = label_stems - image_stems

        if missing_labels:
            print_warning(f"Images without labels: {len(missing_labels)}")
            self.stats['missing_labels'].extend(
                [(split_name, stem) for stem in missing_labels]
            )

        if extra_labels:
            print_warning(f"Labels without images: {len(extra_labels)}")

        # Analyze label files
        for label_file in labels:
            self._analyze_yolo_label(label_file, split_name)

        if len(missing_labels) == 0 and len(extra_labels) == 0:
            print_success(f"All images have matching labels")

    def _analyze_yolo_label(self, label_path, split_name):
        """
        Analyze a single YOLO label file
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                self.stats['empty_labels'].append((split_name, label_path.stem))
                return

            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()

                if len(parts) != 5:
                    self.stats['malformed_annotations'].append(
                        (split_name, label_path.name, line_num, f"Expected 5 values, got {len(parts)}")
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    # Track class distribution
                    self.stats['class_distribution'][class_id] += 1

                    # Track bbox statistics
                    self.stats['bbox_stats']['widths'].append(w)
                    self.stats['bbox_stats']['heights'].append(h)
                    self.stats['bbox_stats']['areas'].append(w * h)

                    # Check for invalid values
                    if class_id < 0 or class_id >= self.num_classes:
                        self.stats['malformed_annotations'].append(
                            (split_name, label_path.name, line_num, f"Invalid class_id: {class_id}")
                        )

                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        self.stats['malformed_annotations'].append(
                            (split_name, label_path.name, line_num, f"Coords out of range: {x:.3f},{y:.3f},{w:.3f},{h:.3f}")
                        )

                except ValueError as e:
                    self.stats['malformed_annotations'].append(
                        (split_name, label_path.name, line_num, f"Parse error: {e}")
                    )

        except Exception as e:
            self.stats['malformed_annotations'].append(
                (split_name, label_path.name, 0, f"File read error: {e}")
            )

    def inspect_coco(self):
        """
        Inspect COCO format dataset
        """
        print_header("INSPECTING COCO FORMAT")

        # Find JSON annotation files
        json_files = list(self.dataset_path.rglob('*.json'))

        for json_file in json_files:
            if 'annotations' in json_file.name.lower() or 'coco' in json_file.name.lower():
                self._analyze_coco_json(json_file)
                break
        else:
            # Analyze the first JSON file found
            if json_files:
                self._analyze_coco_json(json_files[0])

    def _analyze_coco_json(self, json_path):
        """
        Analyze a COCO format JSON file
        """
        print_info(f"Analyzing: {json_path.name}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract categories (classes)
        categories = data.get('categories', [])
        self.classes = [cat['name'] for cat in categories]
        self.num_classes = len(self.classes)

        print_info(f"Number of classes: {self.num_classes}")
        print_info(f"Class names: {self.classes}")

        # Extract images
        images = data.get('images', [])
        print_info(f"Total images: {len(images)}")

        # Extract annotations
        annotations = data.get('annotations', [])
        print_info(f"Total annotations: {len(annotations)}")

        # Count annotations per class
        cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
        for ann in annotations:
            cat_id = ann.get('category_id', -1)
            self.stats['class_distribution'][cat_id] += 1

        # Print class distribution
        print(f"\n{CYAN}Class Distribution:{RESET}")
        for cat_id, count in sorted(self.stats['class_distribution'].items()):
            class_name = cat_id_to_name.get(cat_id, f"Unknown({cat_id})")
            print_info(f"  {class_name}: {count} annotations")

    def inspect_pascal_voc(self):
        """
        Inspect Pascal VOC format dataset
        """
        print_header("INSPECTING PASCAL VOC FORMAT")

        xml_files = list(self.dataset_path.rglob('*.xml'))

        print_info(f"Found {len(xml_files)} XML annotation files")

        # Analyze a sample of XML files to detect classes
        classes_found = set()

        for xml_file in xml_files[:100]:  # Sample first 100
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None:
                        classes_found.add(name_elem.text)

            except Exception as e:
                self.stats['malformed_annotations'].append(
                    ('unknown', xml_file.name, 0, str(e))
                )

        self.classes = sorted(list(classes_found))
        self.num_classes = len(self.classes)

        print_info(f"Detected {self.num_classes} classes: {self.classes}")

        # Count annotations per class
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None:
                        class_name = name_elem.text
                        class_id = self.classes.index(class_name)
                        self.stats['class_distribution'][class_id] += 1

            except Exception as e:
                pass

        print(f"\n{CYAN}Class Distribution:{RESET}")
        for class_id, count in sorted(self.stats['class_distribution'].items()):
            print_info(f"  {self.classes[class_id]}: {count} annotations")

    def generate_report(self):
        """
        Generate a comprehensive report
        """
        print_header("DATASET INSPECTION REPORT")

        # Basic info
        print(f"\n{CYAN}Dataset Information:{RESET}")
        print_info(f"Path: {self.dataset_path}")
        print_info(f"Format: {self.format}")
        print_info(f"Number of classes: {self.num_classes}")
        print_info(f"Classes: {self.classes}")

        # Split statistics
        print(f"\n{CYAN}Split Statistics:{RESET}")
        total_images = 0
        total_labels = 0

        for split_name in self.stats['images_per_split']:
            img_count = self.stats['images_per_split'][split_name]
            lbl_count = self.stats['labels_per_split'][split_name]
            total_images += img_count
            total_labels += lbl_count

            match_status = "MATCH" if img_count == lbl_count else f"MISMATCH ({img_count - lbl_count} diff)"
            print_info(f"{split_name.upper():8s}: {img_count:5d} images, {lbl_count:5d} labels [{match_status}]")

        print_info(f"{'TOTAL':8s}: {total_images:5d} images, {total_labels:5d} labels")

        # Class distribution
        print(f"\n{CYAN}Class Distribution (Annotations per Class):{RESET}")
        total_annotations = sum(self.stats['class_distribution'].values())

        for class_id in range(self.num_classes):
            count = self.stats['class_distribution'].get(class_id, 0)
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"Class_{class_id}"
            bar = '█' * int(percentage / 2)
            print(f"  {class_name:20s}: {count:5d} ({percentage:5.1f}%) {bar}")

        # Issues found
        print(f"\n{CYAN}Issues Found:{RESET}")

        if self.stats['missing_labels']:
            print_warning(f"Images without labels: {len(self.stats['missing_labels'])}")
            if len(self.stats['missing_labels']) <= 10:
                for split, stem in self.stats['missing_labels']:
                    print(f"    - {split}/{stem}")
        else:
            print_success("No images missing labels")

        if self.stats['empty_labels']:
            print_warning(f"Empty label files: {len(self.stats['empty_labels'])}")
            if len(self.stats['empty_labels']) <= 10:
                for split, stem in self.stats['empty_labels']:
                    print(f"    - {split}/{stem}")
        else:
            print_success("No empty label files")

        if self.stats['malformed_annotations']:
            print_warning(f"Malformed annotations: {len(self.stats['malformed_annotations'])}")
            for split, filename, line, error in self.stats['malformed_annotations'][:10]:
                print(f"    - {split}/{filename}:{line} - {error}")
        else:
            print_success("No malformed annotations")

        # Bounding box statistics
        if self.stats['bbox_stats']['widths']:
            print(f"\n{CYAN}Bounding Box Statistics:{RESET}")
            widths = np.array(self.stats['bbox_stats']['widths'])
            heights = np.array(self.stats['bbox_stats']['heights'])
            areas = np.array(self.stats['bbox_stats']['areas'])

            print_info(f"Width:  min={widths.min():.3f}, max={widths.max():.3f}, mean={widths.mean():.3f}")
            print_info(f"Height: min={heights.min():.3f}, max={heights.max():.3f}, mean={heights.mean():.3f}")
            print_info(f"Area:   min={areas.min():.4f}, max={areas.max():.4f}, mean={areas.mean():.4f}")

    def plot_class_distribution(self, save_path=None):
        """
        Plot class distribution as a bar chart
        """
        if not self.stats['class_distribution']:
            print_warning("No class distribution data to plot")
            return

        class_names = [self.classes[i] if i < len(self.classes) else f"Class_{i}"for i in range(self.num_classes)]
        counts = [self.stats['class_distribution'].get(i, 0) for i in range(self.num_classes)]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, counts, color='steelblue', edgecolor='black')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Annotations', fontsize=12)
        plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print_success(f"Plot saved to: {save_path}")

        plt.show()

    def visualize_samples(self, num_samples=5, split_name='train'):
        """
        Visualize random images with their bounding boxes
        """
        print_header(f"VISUALIZING SAMPLE IMAGES FROM {split_name.upper()}")

        if split_name not in self.splits:
            print_error(f"Split '{split_name}' not found. Available: {list(self.splits.keys())}")
            return

        split_path = self.splits[split_name]
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'

        if not images_dir.exists():
            images_dir = split_path

        # Get all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in images_dir.rglob('*') if f.suffix.lower() in image_extensions]

        if not images:
            print_error(f"No images found in {images_dir}")
            return

        # Select random samples
        samples = random.sample(images, min(num_samples, len(images)))

        # Create figure
        fig, axes = plt.subplots(1, len(samples), figsize=(5*len(samples), 5))
        if len(samples) == 1:
            axes = [axes]

        # Color map for classes
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

        for ax, img_path in zip(axes, samples):
            # Load image
            img = Image.open(img_path)
            img_array = np.array(img)

            ax.imshow(img_array)
            ax.set_title(f"{img_path.name[:30]}...", fontsize=9)

            # Load corresponding label
            label_path = labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                img_w, img_h = img.size

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Convert normalized coordinates to pixel coordinates
                    x1 = (x_center - width/2) * img_w
                    y1 = (y_center - height/2) * img_h
                    w = width * img_w
                    h = height * img_h

                    # Draw bounding box
                    color = colors[class_id % len(colors)]
                    rect = patches.Rectangle(
                        (x1, y1), w, h,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add class label
                    class_name = self.classes[class_id] if class_id < len(self.classes) else f"Class_{class_id}"
                    ax.text(x1, y1 - 5, class_name, fontsize=8,
                           bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'),
                           color='white', fontweight='bold')

            ax.axis('off')

        plt.tight_layout()
        plt.suptitle(f"Sample Images with Bounding Boxes ({split_name})", fontsize=12, y=1.02)
        plt.show()


def main():
    """
    Main function to run the dataset inspection
    """
    # ============================================
    # CONFIGURATION - Modify this path as needed
    # ============================================
    DATASET_PATH = r"F:\Research Project\Mosquito Detection Dataset.v4i.yolov11"

    # ============================================

    print_header("OBJECT DETECTION DATASET INSPECTOR")
    print(f"\n{CYAN}Dataset Path: {DATASET_PATH}{RESET}")

    # Create inspector
    inspector = DatasetInspector(DATASET_PATH)

    # Step 1: Detect format
    fmt = inspector.detect_format()

    if fmt is None:
        print_error("Cannot proceed without detecting format")
        return

    # Step 2: Inspect based on format
    if fmt == 'YOLO':
        inspector.inspect_yolo()
    elif fmt == 'COCO':
        inspector.inspect_coco()
    elif fmt == 'Pascal_VOC':
        inspector.inspect_pascal_voc()

    # Step 3: Generate report
    inspector.generate_report()

    # Step 4: Plot class distribution
    print(f"\n{CYAN}Generating class distribution plot...{RESET}")
    inspector.plot_class_distribution(save_path="class_distribution.png")

    # Step 5: Visualize sample images
    print(f"\n{CYAN}Visualizing sample images with bounding boxes...{RESET}")
    inspector.visualize_samples(num_samples=5, split_name='train')

    # Step 6: Print final checklist
    print_header("VERIFICATION CHECKLIST")

    checklist = [
        ("Dataset recognized", inspector.format is not None),
        ("Classes identified", inspector.num_classes > 0),
        ("Annotation format identified", inspector.format in ['YOLO', 'COCO', 'Pascal_VOC']),
        ("Train/val/test splits identified", len(inspector.splits) > 0),
        ("No major missing-label problems", len(inspector.stats['missing_labels']) == 0),
        ("No malformed annotations", len(inspector.stats['malformed_annotations']) == 0),
    ]

    all_passed = True
    for item, passed in checklist:
        status = f"{GREEN}[X]{RESET}" if passed else f"{RED}[ ]{RESET}"
        print(f"  {status} {item}")
        if not passed:
            all_passed = False

    print(f"\n{'='*70}")
    if all_passed:
        print(f"{GREEN}ALL CHECKS PASSED - Dataset is ready for training!{RESET}")
    else:
        print(f"{YELLOW}Some checks failed - Review issues above{RESET}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
