"""
Master Setup Guide for Mosquito Detection Project
==================================================
STAGE 3: Complete Dataset Preparation

This script is your guide to setting up the project from start to finish.
Follow these steps in order.
"""

def print_guide():
    guide = """
╔════════════════════════════════════════════════════════════════╗
║        MOSQUITO DETECTION PROJECT - STAGE 3 SETUP GUIDE       ║
╚════════════════════════════════════════════════════════════════╝

PROJECT GOAL:
  Train a YOLOv11 model to detect and classify 6 mosquito species

DATASET LOCATION:
  Source: F:\\Research Project\\Mosquito Detection Dataset.v4i.yolov11
  Target: E:\\Claude local session\\mosquito-detection

WHAT WE'RE DOING:
  Converting a Roboflow-formatted dataset into a clean project structure
  that's ready for YOLO training.

═══════════════════════════════════════════════════════════════════

STEP-BY-STEP INSTRUCTIONS:

STEP 1: Run the Project Setup Script
────────────────────────────────────
  Purpose: Create the project structure and copy all files

  Command:
    cd "E:\\Claude local session"
    python setup_project_structure.py

  What it does:
    ✓ Creates directory structure
    ✓ Copies images and labels from source dataset
    ✓ Creates data.yaml configuration file
    ✓ Generates README.md
    ✓ Prints copy summary

  Expected output:
    ✓ CREATING PROJECT DIRECTORY STRUCTURE
    ✓ COPYING DATASET FILES
    ✓ CREATING data.yaml
    ✓ SETUP COMPLETE!

  Time: ~2-5 minutes (copying 7,672 files)

═══════════════════════════════════════════════════════════════════

STEP 2: Verify the Dataset
────────────────────────────
  Purpose: Check for issues and ensure data integrity

  Command:
    python verify_prepared_dataset.py

  What it checks:
    ✓ Directory structure is correct
    ✓ data.yaml is valid
    ✓ All images have corresponding labels
    ✓ No data leakage between splits
    ✓ All annotations are valid YOLO format
    ✓ Class IDs are within valid range [0, 5]
    ✓ Coordinates are normalized [0, 1]

  Expected output (all GREEN):
    ✓ Directory structure
    ✓ data.yaml
    ✓ Image-label pairs
    ✓ Data leakage
    ✓ Annotations

  Time: ~1 minute

═══════════════════════════════════════════════════════════════════

STEP 3: Visualize the Dataset
───────────────────────────────
  Purpose: Visually verify that bounding boxes are correct

  Command:
    python visualize_samples.py

  What it does:
    ✓ Loads random images from train/val/test splits
    ✓ Draws bounding boxes with class labels
    ✓ Shows statistics about samples
    ✓ Displays images in matplotlib grid
    ✓ Optional: OpenCV one-by-one view

  What to look for:
    ✓ Boxes tightly enclose mosquitoes
    ✓ Class labels are correct (aegypti, albopictus, etc.)
    ✓ No boxes outside image boundaries
    ✓ Reasonable distribution across splits

  Time: ~30 seconds

═══════════════════════════════════════════════════════════════════

DATASET STRUCTURE CREATED:

E:\\Claude local session\\mosquito-detection\\
│
├── dataset/
│   ├── images/
│   │   ├── train/      (6,633 images)
│   │   ├── val/        (670 images)
│   │   └── test/       (369 images)
│   └── labels/
│       ├── train/      (6,633 labels)
│       ├── val/        (670 labels)
│       └── test/       (369 labels)
│
├── data.yaml           (Main configuration file)
├── models/             (Where trained models will be saved)
├── results/            (Where training results will be saved)
├── README.md           (Project documentation)
│
├── setup_project_structure.py    (Setup script)
├── verify_prepared_dataset.py    (Verification script)
└── visualize_samples.py          (Visualization script)

═══════════════════════════════════════════════════════════════════

DATA.YAML CONFIGURATION:

The data.yaml file contains:
  - Paths to train/val/test images
  - Number of classes: 6
  - Class names:
    0: aegypti           (Aedes aegypti - Yellow fever mosquito)
    1: albopictus        (Aedes albopictus - Asian tiger mosquito)
    2: anopheles         (Anopheles - Malaria vector)
    3: culex             (Culex - Common house mosquito)
    4: culiseta          (Culiseta - Large mosquito)
    5: japonicus-koreicus (Aedes japonicus/koreicus - Bush mosquito)

  - Image size: 640x640 pixels
  - Format: YOLO (normalized bounding boxes)

═══════════════════════════════════════════════════════════════════

DATASET STATISTICS:

Total Images: 7,672
  - Train: 6,633 (86.4%)
  - Val:   670   (8.7%)
  - Test:  369   (4.8%)

Total Annotations: One or more per image
Image Format: JPEG
Image Size: 640×640 pixels (pre-resized by Roboflow)
Annotation Format: YOLO txt (normalized coordinates)

═══════════════════════════════════════════════════════════════════

KEY CONCEPTS TO REMEMBER:

1. NORMALIZED COORDINATES:
   All coordinates in YOLO format are normalized to [0, 1]
   - 0.0 = left/top edge
   - 0.5 = middle
   - 1.0 = right/bottom edge

2. CENTER-BASED BOUNDING BOXES:
   Format: class_id x_center y_center width height
   - x_center, y_center = center point of the box
   - width, height = size relative to image

3. CLASS IDs:
   0-indexed, starting from 0
   Your dataset: 0-5 (6 classes)

4. NO DATA LEAKAGE:
   Train/Val/Test splits are completely separate
   No image appears in multiple splits

═══════════════════════════════════════════════════════════════════

VERIFICATION CHECKLIST:

After running all 3 scripts, verify:

[ ] Project directory created: E:\\Claude local session\\mosquito-detection
[ ] All subdirectories exist (dataset, images, labels, models, results)
[ ] data.yaml created and points to correct paths
[ ] All 7,672 images copied to correct directories
[ ] All 7,672 label files copied to correct directories
[ ] verify_prepared_dataset.py shows all GREEN checks
[ ] No data leakage detected
[ ] visualize_samples.py shows correct bounding boxes
[ ] Bounding boxes enclose mosquitoes properly
[ ] Class labels are readable and correct

═══════════════════════════════════════════════════════════════════

TROUBLESHOOTING:

If setup_project_structure.py fails:
  - Check that source path exists: F:\\Research Project\\Mosquito Detection Dataset.v4i.yolov11
  - Ensure disk has enough space (~5GB for full dataset copy)
  - Try running from Command Prompt with admin privileges

If verify_prepared_dataset.py shows errors:
  - Check that setup completed successfully
  - Look for missing image-label pairs (run again if interrupted)
  - Check data.yaml for correct paths

If visualize_samples.py doesn't show images:
  - Ensure OpenCV/matplotlib are installed: pip install opencv-python matplotlib
  - Check that images were copied successfully
  - Verify label files are readable

═══════════════════════════════════════════════════════════════════

NEXT STEPS (STAGE 4):

Once all verification passes, you're ready for:
  1. Create train.py script
  2. Configure YOLOv11 training parameters
  3. Train the first model
  4. Evaluate model performance

═══════════════════════════════════════════════════════════════════

QUICK REFERENCE - RUN THESE COMMANDS IN ORDER:

1. python setup_project_structure.py
2. python verify_prepared_dataset.py
3. python visualize_samples.py

Then: cd E:\\Claude local session\\mosquito-detection
      (You'll train your model here in Stage 4)

═══════════════════════════════════════════════════════════════════
    """
    print(guide)


if __name__ == "__main__":
    print_guide()

    # Ask user if they want to proceed
    print("\n" + "="*60)
    print("READY TO BEGIN?")
    print("="*60)
    print("\nMake sure you have:")
    print("  [ ] Python 3.8+ installed")
    print("  [ ] PyYAML installed: pip install pyyaml")
    print("  [ ] OpenCV installed: pip install opencv-python")
    print("  [ ] Matplotlib installed: pip install matplotlib")
    print("  [ ] NumPy installed: pip install numpy")
    print("\nRun the scripts in order:")
    print("  1. python setup_project_structure.py")
    print("  2. python verify_prepared_dataset.py")
    print("  3. python visualize_samples.py")
    print("\n" + "="*60)
