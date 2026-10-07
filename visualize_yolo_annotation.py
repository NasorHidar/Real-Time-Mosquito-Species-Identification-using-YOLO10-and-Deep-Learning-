"""
YOLO Annotation Visualizer
==========================
Stage 2: Understanding YOLO annotations through visualization

This script:
1. Reads one image
2. Reads its YOLO label file
3. Converts normalized YOLO coordinates to pixel coordinates
4. Draws the bounding box using OpenCV
5. Displays class name and coordinates
"""

import cv2
import numpy as np
from pathlib import Path
import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset paths
DATASET_PATH = Path(r"F:\Research Project\Mosquito Detection Dataset.v4i.yolov11")

# Select a specific image to visualize (set to None to use a random image)
SPECIFIC_IMAGE = None  # Or put a filename like "0000c8c4-e87a-44b8-84d4-8bebcf75645c_jpeg.rf.424cbaa949db1b40b3cdf7187efb076c.jpg"

# Split to use (train, valid, or test)
SPLIT = "train"

# Display settings
WINDOW_NAME = "YOLO Annotation Visualizer"
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# =============================================================================


def load_class_names(dataset_path):
    """Load class names from data.yaml"""
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    return []


def parse_yolo_annotation(label_path, class_names):
    """
    Parse a YOLO label file and return list of annotations.

    Each annotation is a dict with:
        - class_id: int
        - class_name: str
        - x_center, y_center, width, height: float (normalized)
    """
    annotations = []

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            print(f"[WARNING] Line {line_num}: Expected 5 values, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            annotations.append({
                'class_id': class_id,
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

        except ValueError as e:
            print(f"[WARNING] Line {line_num}: Parse error - {e}")

    return annotations


def validate_annotation(ann, num_classes):
    """
    Validate a YOLO annotation for common errors.

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check class_id
    if ann['class_id'] < 0 or ann['class_id'] >= num_classes:
        errors.append(f"Invalid class_id {ann['class_id']} (must be 0-{num_classes-1})")

    # Check coordinates are in [0, 1]
    for coord_name in ['x_center', 'y_center', 'width', 'height']:
        value = ann[coord_name]
        if value < 0:
            errors.append(f"{coord_name} is negative: {value}")
        elif value > 1:
            errors.append(f"{coord_name} > 1: {value}")

    # Check width and height are positive
    if ann['width'] <= 0:
        errors.append(f"Width must be > 0, got {ann['width']}")
    if ann['height'] <= 0:
        errors.append(f"Height must be > 0, got {ann['height']}")

    return errors


def yolo_to_pixel_coords(ann, img_width, img_height):
    """
    Convert YOLO normalized coordinates to pixel coordinates.

    Returns:
        x1, y1, x2, y2: Corner coordinates in pixels
        x_center, y_center: Center coordinates in pixels
        box_width, box_height: Box dimensions in pixels
    """
    # Convert normalized to pixels
    x_center_px = ann['x_center'] * img_width
    y_center_px = ann['y_center'] * img_height
    box_width_px = ann['width'] * img_width
    box_height_px = ann['height'] * img_height

    # Calculate corners
    x1 = x_center_px - box_width_px / 2
    y1 = y_center_px - box_height_px / 2
    x2 = x_center_px + box_width_px / 2
    y2 = y_center_px + box_height_px / 2

    return {
        'x1': int(x1),
        'y1': int(y1),
        'x2': int(x2),
        'y2': int(y2),
        'x_center_px': int(x_center_px),
        'y_center_px': int(y_center_px),
        'box_width_px': int(box_width_px),
        'box_height_px': int(box_height_px)
    }


def draw_annotation(img, ann, px_coords, color):
    """
    Draw a bounding box and label on an image.
    """
    x1, y1 = px_coords['x1'], px_coords['y1']
    x2, y2 = px_coords['x2'], px_coords['y2']

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Draw center point
    cx, cy = px_coords['x_center_px'], px_coords['y_center_px']
    cv2.circle(img, (cx, cy), 5, color, -1)

    # Draw label background
    label = f"{ann['class_name']} (id={ann['class_id']})"
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

    # Draw label text
    cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)


def print_annotation_info(ann, px_coords, errors):
    """
    Print detailed information about an annotation.
    """
    print(f"\n{'='*60}")
    print(f"ANNOTATION: {ann['class_name']}")
    print(f"{'='*60}")

    print(f"\n[YOLO FORMAT - Normalized Coordinates]")
    print(f"  class_id:    {ann['class_id']}")
    print(f"  x_center:    {ann['x_center']:.6f}")
    print(f"  y_center:    {ann['y_center']:.6f}")
    print(f"  width:       {ann['width']:.6f}")
    print(f"  height:      {ann['height']:.6f}")

    print(f"\n[PIXEL COORDINATES]")
    print(f"  Center:      ({px_coords['x_center_px']}, {px_coords['y_center_px']})")
    print(f"  Box size:    {px_coords['box_width_px']} x {px_coords['box_height_px']} pixels")
    print(f"  Top-left:    ({px_coords['x1']}, {px_coords['y1']})")
    print(f"  Bottom-right: ({px_coords['x2']}, {px_coords['y2']})")

    if errors:
        print(f"\n[ERRORS DETECTED]")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n[VALIDATION: PASSED]")


def main():
    """Main function to visualize YOLO annotations."""

    print(f"\n{'='*60}")
    print(f"YOLO ANNOTATION VISUALIZER")
    print(f"{'='*60}")

    # Load class names
    class_names = load_class_names(DATASET_PATH)
    num_classes = len(class_names)

    if not class_names:
        print("[ERROR] Could not load class names from data.yaml")
        return

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Define paths
    images_dir = DATASET_PATH / SPLIT / "images"
    labels_dir = DATASET_PATH / SPLIT / "labels"

    # Get image file
    if SPECIFIC_IMAGE:
        image_path = images_dir / SPECIFIC_IMAGE
        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            return
    else:
        # Get first image
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            print(f"[ERROR] No images found in {images_dir}")
            return
        image_path = image_files[0]

    print(f"\nImage: {image_path.name}")

    # Get label file
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        print(f"[ERROR] Label file not found: {label_path}")
        return

    print(f"Label: {label_path.name}")

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    img_height, img_width = img.shape[:2]
    print(f"Image size: {img_width} x {img_height} pixels")

    # Parse annotations
    annotations = parse_yolo_annotation(label_path, class_names)
    print(f"\nNumber of objects: {len(annotations)}")

    if not annotations:
        print("[WARNING] No annotations found in label file")
        return

    # Generate colors for each class
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Process each annotation
    all_valid = True
    for i, ann in enumerate(annotations):
        # Validate
        errors = validate_annotation(ann, num_classes)

        # Convert to pixel coordinates
        px_coords = yolo_to_pixel_coords(ann, img_width, img_height)

        # Print info
        print_annotation_info(ann, px_coords, errors)

        # Draw on image
        color = colors[ann['class_id'] % len(colors)]
        draw_annotation(img, ann, px_coords, color)

        if errors:
            all_valid = False

    # Print coordinate conversion example
    if annotations:
        ann = annotations[0]
        print(f"\n{'='*60}")
        print(f"COORDINATE CONVERSION EXAMPLE")
        print(f"{'='*60}")
        print(f"\nGiven YOLO annotation:")
        print(f"  {ann['class_id']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}")
        print(f"\nImage size: {img_width} x {img_height}")
        print(f"\nConversion formulas:")
        print(f"  x_center_px = x_center * img_width")
        print(f"              = {ann['x_center']} * {img_width} = {ann['x_center'] * img_width:.1f}")
        print(f"  y_center_px = y_center * img_height")
        print(f"              = {ann['y_center']} * {img_height} = {ann['y_center'] * img_height:.1f}")
        print(f"  x1 = x_center_px - width_px / 2 = {ann['x_center'] * img_width:.1f} - {ann['width'] * img_width / 2:.1f} = {(ann['x_center'] - ann['width']/2) * img_width:.1f}")
        print(f"  y1 = y_center_px - height_px / 2 = {ann['y_center'] * img_height:.1f} - {ann['height'] * img_height / 2:.1f} = {(ann['y_center'] - ann['height']/2) * img_height:.1f}")

    # Add info overlay on image
    info_text = f"Image: {image_path.name[:30]}... | Objects: {len(annotations)}"
    cv2.putText(img, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show image
    print(f"\n{'='*60}")
    print(f"Displaying image. Press any key to close.")
    print(f"{'='*60}")

    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
