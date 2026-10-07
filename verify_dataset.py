"""
YOLO Dataset Verification Script
Stage 1: Verify that a mosquito detection dataset is suitable for YOLO training

This script checks:
1. Folder structure (train/valid/test splits exist)
2. Image and label counts match
3. Label format is valid YOLO txt format
4. No corrupted images
5. Class IDs are within valid range
6. Basic statistics about the dataset
"""

import os
import yaml
from pathlib import Path
from PIL import Image
import numpy as np

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print a colored header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}[OK] {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}[ERROR] {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}[WARNING] {text}{RESET}")

def verify_folder_structure(dataset_path):
    """Check if train/valid/test folders exist with images and labels subdirectories"""
    print_header("CHECKING FOLDER STRUCTURE")

    required_splits = ['train', 'valid', 'test']
    all_exist = True

    for split in required_splits:
        split_path = dataset_path / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'

        if split_path.exists():
            print_success(f"{split}/ folder exists")
        else:
            print_error(f"{split}/ folder NOT FOUND")
            all_exist = False
            continue

        if images_path.exists():
            print_success(f"  -> {split}/images/ exists")
        else:
            print_error(f"  -> {split}/images/ NOT FOUND")
            all_exist = False

        if labels_path.exists():
            print_success(f"  -> {split}/labels/ exists")
        else:
            print_error(f"  -> {split}/labels/ NOT FOUND")
            all_exist = False

    return all_exist

def verify_data_yaml(dataset_path):
    """Check if data.yaml exists and is properly formatted"""
    print_header("CHECKING data.yaml")

    yaml_path = dataset_path / 'data.yaml'

    if not yaml_path.exists():
        print_error("data.yaml NOT FOUND")
        return None

    print_success("data.yaml exists")

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Check required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field in data:
                print_success(f"  -> '{field}' field present")
            else:
                print_error(f"  -> '{field}' field MISSING")
                return None

        # Print dataset info
        print(f"\n{BLUE}Dataset Configuration:{RESET}")
        print(f"  Number of classes (nc): {data['nc']}")
        print(f"  Class names: {data['names']}")

        return data

    except Exception as e:
        print_error(f"Error reading data.yaml: {e}")
        return None

def count_files(directory, extensions):
    """Count files with specific extensions in a directory"""
    if not directory.exists():
        return 0
    return len([f for f in directory.iterdir() if f.suffix.lower() in extensions])

def verify_image_label_pairs(dataset_path):
    """Check that each image has a corresponding label file"""
    print_header("CHECKING IMAGE-LABEL PAIRS")

    splits = ['train', 'valid', 'test']
    image_extensions = ['.jpg', '.jpeg', '.png']

    stats = {}
    all_matched = True

    for split in splits:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'

        if not images_dir.exists() or not labels_dir.exists():
            print_warning(f"Skipping {split} (directory missing)")
            continue

        # Count images and labels
        image_count = count_files(images_dir, image_extensions)
        label_count = count_files(labels_dir, ['.txt'])

        stats[split] = {'images': image_count, 'labels': label_count}

        print(f"\n{split.upper()}:")
        print(f"  Images: {image_count}")
        print(f"  Labels: {label_count}")

        if image_count == label_count:
            print_success(f"  Match! {image_count} images = {label_count} labels")
        else:
            print_error(f"  Mismatch! {image_count} images ≠ {label_count} labels")
            all_matched = False

    return stats, all_matched

def verify_label_format(dataset_path, num_classes, samples_to_check=10):
    """Verify that label files are in proper YOLO format"""
    print_header("CHECKING LABEL FORMAT")

    splits = ['train', 'valid', 'test']
    format_valid = True

    for split in splits:
        labels_dir = dataset_path / split / 'labels'

        if not labels_dir.exists():
            continue

        label_files = list(labels_dir.glob('*.txt'))

        if not label_files:
            print_warning(f"No label files found in {split}")
            continue

        # Check a sample of labels
        sample_size = min(samples_to_check, len(label_files))
        samples = np.random.choice(label_files, sample_size, replace=False)

        print(f"\n{split.upper()}: Checking {sample_size} random label files...")

        errors = 0

        for label_file in samples:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                # Skip empty files (images with no objects)
                if not lines:
                    continue

                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()

                    # YOLO format: class_id center_x center_y width height
                    if len(parts) != 5:
                        print_error(f"  {label_file.name}:{line_num} - Expected 5 values, got {len(parts)}")
                        errors += 1
                        format_valid = False
                        continue

                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])

                        # Check class ID range
                        if class_id < 0 or class_id >= num_classes:
                            print_error(f"  {label_file.name}:{line_num} - Invalid class_id {class_id} (must be 0-{num_classes-1})")
                            errors += 1
                            format_valid = False

                        # Check coordinate ranges (should be 0-1)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            print_error(f"  {label_file.name}:{line_num} - Coordinates out of range (must be 0-1)")
                            errors += 1
                            format_valid = False

                    except ValueError as e:
                        print_error(f"  {label_file.name}:{line_num} - Cannot parse values: {e}")
                        errors += 1
                        format_valid = False

            except Exception as e:
                print_error(f"  Error reading {label_file.name}: {e}")
                errors += 1
                format_valid = False

        if errors == 0:
            print_success(f"  All {sample_size} checked labels have valid format")
        else:
            print_error(f"  Found {errors} format errors in {split}")

    return format_valid

def verify_images(dataset_path, samples_to_check=10):
    """Check for corrupted images"""
    print_header("CHECKING IMAGE FILES")

    splits = ['train', 'valid', 'test']
    all_valid = True

    for split in splits:
        images_dir = dataset_path / split / 'images'

        if not images_dir.exists():
            continue

        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))

        if not image_files:
            print_warning(f"No images found in {split}")
            continue

        # Check a sample
        sample_size = min(samples_to_check, len(image_files))
        samples = np.random.choice(image_files, sample_size, replace=False)

        print(f"\n{split.upper()}: Checking {sample_size} random images...")

        corrupted = 0

        for img_file in samples:
            try:
                img = Image.open(img_file)
                img.verify()  # Check if image is corrupted
            except Exception as e:
                print_error(f"  Corrupted image: {img_file.name}")
                corrupted += 1
                all_valid = False

        if corrupted == 0:
            print_success(f"  All {sample_size} checked images are valid")
        else:
            print_error(f"  Found {corrupted} corrupted images in {split}")

    return all_valid

def print_final_summary(folder_ok, yaml_ok, pairs_ok, format_ok, images_ok):
    """Print final verdict"""
    print_header("FINAL VERDICT")

    checks = [
        ("Folder structure", folder_ok),
        ("data.yaml valid", yaml_ok),
        ("Image-label pairs match", pairs_ok),
        ("Label format valid", format_ok),
        ("Images not corrupted", images_ok)
    ]

    all_passed = all(check[1] for check in checks)

    for check_name, passed in checks:
        if passed:
            print_success(check_name)
        else:
            print_error(check_name)

    print("\n" + "="*60)
    if all_passed:
        print(f"{GREEN}[OK] DATASET IS READY FOR YOLO TRAINING!{RESET}")
    else:
        print(f"{RED}[ERROR] DATASET HAS ISSUES - FIX BEFORE TRAINING{RESET}")
    print("="*60 + "\n")

    return all_passed

def main():
    """Main verification function"""

    # Dataset path (adjust this to your dataset location)
    dataset_path = Path(r"F:\Research Project\Mosquito Detection Dataset.v4i.yolov11")

    print(f"\n{BLUE}Dataset Path: {dataset_path}{RESET}")

    if not dataset_path.exists():
        print_error(f"Dataset path does not exist: {dataset_path}")
        return

    # Run all checks
    folder_ok = verify_folder_structure(dataset_path)

    yaml_data = verify_data_yaml(dataset_path)
    yaml_ok = yaml_data is not None

    num_classes = yaml_data['nc'] if yaml_data else 0

    stats, pairs_ok = verify_image_label_pairs(dataset_path)

    format_ok = verify_label_format(dataset_path, num_classes) if yaml_ok else False

    images_ok = verify_images(dataset_path)

    # Print summary
    dataset_ready = print_final_summary(folder_ok, yaml_ok, pairs_ok, format_ok, images_ok)

    if dataset_ready:
        print(f"{GREEN}Next step: Proceed to Stage 2 - Inspect Images and Annotations{RESET}")
    else:
        print(f"{YELLOW}Fix the issues above before proceeding{RESET}")

if __name__ == "__main__":
    main()
