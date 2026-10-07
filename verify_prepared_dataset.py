"""
Verify Prepared Dataset
=======================
Stage 3: Verification script for the prepared YOLO dataset

This script checks:
1. Directory structure is correct
2. data.yaml is valid
3. All image-label pairs exist
4. No data leakage between splits
5. Class IDs are valid
6. Coordinates are within bounds
"""

import yaml
from pathlib import Path
from collections import Counter, defaultdict
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"E:\Claude local session\mosquito-detection")

# =============================================================================


def check_directory_structure():
    """Verify the directory structure exists"""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)

    required_dirs = [
        "dataset",
        "dataset/images",
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "models",
        "results",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"[OK] {dir_path}")
        else:
            print(f"[ERROR] Missing: {dir_path}")
            all_exist = False

    return all_exist


def check_data_yaml():
    """Verify data.yaml is valid and points to correct locations"""
    print("\n" + "="*60)
    print("CHECKING data.yaml")
    print("="*60)

    yaml_path = PROJECT_ROOT / "data.yaml"

    if not yaml_path.exists():
        print("[ERROR] data.yaml not found")
        return False

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Check required fields
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print(f"[ERROR] Missing fields in data.yaml: {missing_fields}")
            return False

        print(f"[OK] All required fields present")
        print(f"\nConfiguration:")
        print(f"  path: {data['path']}")
        print(f"  train: {data['train']}")
        print(f"  val: {data['val']}")
        print(f"  test: {data.get('test', 'N/A')}")
        print(f"  nc: {data['nc']}")
        print(f"  names: {data['names']}")

        # Verify paths exist
        base_path = Path(data['path'])
        for split_name in ['train', 'val', 'test']:
            if split_name in data:
                split_path = base_path / data[split_name]
                if split_path.exists():
                    print(f"[OK] {split_name} path exists: {split_path}")
                else:
                    print(f"[WARNING] {split_name} path not found: {split_path}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to parse data.yaml: {e}")
        return False


def check_image_label_pairs():
    """Verify all images have corresponding labels and vice versa"""
    print("\n" + "="*60)
    print("CHECKING IMAGE-LABEL PAIRS")
    print("="*60)

    issues = []

    for split in ['train', 'val', 'test']:
        images_dir = PROJECT_ROOT / "dataset" / "images" / split
        labels_dir = PROJECT_ROOT / "dataset" / "labels" / split

        # Get all files
        image_files = {f.stem for f in images_dir.glob('*.jpg')} | \
                     {f.stem for f in images_dir.glob('*.png')} | \
                     {f.stem for f in images_dir.glob('*.jpeg')}
        label_files = {f.stem for f in labels_dir.glob('*.txt')}

        print(f"\n{split.upper()}:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")

        # Check for mismatches
        missing_labels = image_files - label_files
        extra_labels = label_files - image_files

        if missing_labels:
            print(f"  [ERROR] {len(missing_labels)} images without labels")
            issues.append(f"{split}: {len(missing_labels)} images without labels")
            # Show first 5
            for stem in list(missing_labels)[:5]:
                print(f"    - {stem}")
        else:
            print(f"  [OK] All images have labels")

        if extra_labels:
            print(f"  [WARNING] {len(extra_labels)} labels without images")
            issues.append(f"{split}: {len(extra_labels)} extra labels")
        else:
            print(f"  [OK] No extra labels")

    return len(issues) == 0


def check_data_leakage():
    """Check for duplicate files across train/val/test splits"""
    print("\n" + "="*60)
    print("CHECKING FOR DATA LEAKAGE")
    print("="*60)

    splits = ['train', 'val', 'test']
    image_hashes = defaultdict(list)

    # Calculate hash for each image
    for split in splits:
        images_dir = PROJECT_ROOT / "dataset" / "images" / split
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.png')) + \
                     list(images_dir.glob('*.jpeg'))

        print(f"Hashing {split} images...")
        for img_path in image_files:
            # Calculate MD5 hash
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            image_hashes[img_hash].append((split, img_path.name))

    # Find duplicates across splits
    leakage_found = False
    for img_hash, occurrences in image_hashes.items():
        if len(occurrences) > 1:
            splits_involved = {split for split, _ in occurrences}
            if len(splits_involved) > 1:
                leakage_found = True
                print(f"\n[ERROR] Duplicate image found across splits:")
                for split, filename in occurrences:
                    print(f"  - {split}/{filename}")

    if not leakage_found:
        print("[OK] No data leakage detected")

    return not leakage_found


def check_annotations():
    """Verify all annotations are valid YOLO format"""
    print("\n" + "="*60)
    print("CHECKING ANNOTATIONS")
    print("="*60)

    # Load class info
    yaml_path = PROJECT_ROOT / "data.yaml"
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    num_classes = data['nc']
    class_names = data['names']

    stats = {
        'total_annotations': 0,
        'class_distribution': Counter(),
        'errors': []
    }

    for split in ['train', 'val', 'test']:
        labels_dir = PROJECT_ROOT / "dataset" / "labels" / split
        label_files = list(labels_dir.glob('*.txt'))

        print(f"\nChecking {split} annotations ({len(label_files)} files)...")

        for label_path in label_files:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # Check format
                if len(parts) != 5:
                    stats['errors'].append(
                        f"{split}/{label_path.name}:{line_num} - Expected 5 values, got {len(parts)}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    stats['total_annotations'] += 1
                    stats['class_distribution'][class_id] += 1

                    # Validate class_id
                    if class_id < 0 or class_id >= num_classes:
                        stats['errors'].append(
                            f"{split}/{label_path.name}:{line_num} - Invalid class_id {class_id} (must be 0-{num_classes-1})"
                        )

                    # Validate coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        stats['errors'].append(
                            f"{split}/{label_path.name}:{line_num} - Center coords out of [0,1]: ({x_center:.3f}, {y_center:.3f})"
                        )

                    if not (0 < width <= 1 and 0 < height <= 1):
                        stats['errors'].append(
                            f"{split}/{label_path.name}:{line_num} - Dimensions out of (0,1]: ({width:.3f}, {height:.3f})"
                        )

                except ValueError as e:
                    stats['errors'].append(
                        f"{split}/{label_path.name}:{line_num} - Parse error: {e}"
                    )

    # Print statistics
    print(f"\nTotal annotations: {stats['total_annotations']}")
    print(f"\nClass distribution:")
    for class_id in range(num_classes):
        count = stats['class_distribution'].get(class_id, 0)
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
        print(f"  {class_id}: {class_name:20s} - {count:5d} ({percentage:5.1f}%)")

    if stats['errors']:
        print(f"\n[ERROR] Found {len(stats['errors'])} annotation errors:")
        for error in stats['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
        return False
    else:
        print("\n[OK] All annotations are valid")
        return True


def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    print(f"Project: {PROJECT_ROOT}")

    if not PROJECT_ROOT.exists():
        print(f"\n[ERROR] Project directory not found: {PROJECT_ROOT}")
        print("Please run setup_project_structure.py first")
        return

    # Run checks
    checks = {
        'Directory structure': check_directory_structure(),
        'data.yaml': check_data_yaml(),
        'Image-label pairs': check_image_label_pairs(),
        'Data leakage': check_data_leakage(),
        'Annotations': check_annotations(),
    }

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_passed = True
    for check_name, passed in checks.items():
        status = "[OK]" if passed else "[FAILED]"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL CHECKS PASSED!")
        print("Dataset is ready for training.")
    else:
        print("SOME CHECKS FAILED")
        print("Please fix the issues above before training.")
    print("="*60)


if __name__ == "__main__":
    main()
