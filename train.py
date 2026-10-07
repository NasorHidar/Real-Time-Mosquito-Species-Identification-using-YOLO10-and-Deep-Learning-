"""
Mosquito Detection YOLO Training Script
=========================================
Stage 4: Train the first baseline YOLO model

This script:
1. Checks Ultralytics version
2. Detects GPU/CPU availability
3. Loads the YOLO model
4. Trains on your mosquito dataset
5. Saves the best model
6. Saves training results
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION - Modify these values for your experiment
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_YAML = PROJECT_ROOT / "data.yaml"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Training parameters - BASELINE (simple, fast, good for learning)
MODEL_NAME = "yolo11n"  # YOLOv11 Nano (smallest, fastest for learning)
IMAGE_SIZE = 640        # Input image size (pixels) - moderate for baseline
BATCH_SIZE = 16         # Images processed at once - adjust based on GPU
EPOCHS = 30             # Number of training epochs - baseline start
LEARNING_RATE = 0.01    # Initial learning rate
WEIGHT_DECAY = 0.0005   # L2 regularization to prevent overfitting
PATIENCE = 10           # Early stopping patience (epochs without improvement)

# Augmentation settings (balanced for baseline)
AUGMENT = True          # Enable data augmentation
FLIP_UD = 0.0           # Vertical flip probability
FLIP_LR = 0.5           # Horizontal flip probability

# Hardware settings
DEVICE = "auto"         # "auto" = use GPU if available, "cpu" = force CPU

# Save settings
SAVE_BEST = True        # Save only best model
SAVE_PERIOD = -1        # Save checkpoint every N epochs (-1 = only best)
SAVE_JSON = True        # Save results to JSON

# =============================================================================


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_ultralytics_version():
    """Check installed Ultralytics version"""
    print_section("CHECKING ULTRALYTICS VERSION")

    try:
        import ultralytics
        version = ultralytics.__version__
        print(f"[OK] Ultralytics version: {version}")
        return version
    except ImportError:
        print("[ERROR] Ultralytics not installed!")
        print("        Run: pip install ultralytics")
        sys.exit(1)


def check_gpu_available():
    """Detect and report GPU/CUDA availability"""
    print_section("CHECKING GPU AVAILABILITY")

    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            print(f"[OK] CUDA is available!")
            print(f"     PyTorch version: {torch.__version__}")
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     GPU count: {torch.cuda.device_count()}")

            if torch.cuda.device_count() > 0:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"     GPU: {gpu_name}")
                print(f"     Memory: {gpu_memory:.1f} GB")
                print(f"\n     → Training will use GPU (CUDA)")
        else:
            print("[WARNING] CUDA not available")
            print(f"     PyTorch version: {torch.__version__}")
            print(f"     → Training will use CPU (slower)")

            # Provide Kaggle instructions
            print("\n" + "-"*50)
            print("TO USE GPU, YOU CAN:")
            print("-"*50)
            print("1. Run on Kaggle (free GPU):")
            print("   - Go to kaggle.com")
            print("   - Create new notebook")
            print("   - Upload your project")
            print("   - Enable GPU: Accelerator = GPU P100")
            print("")
            print("2. Run locally with GPU:")
            print("   - Install CUDA: https://developer.nvidia.com/cuda-downloads")
            print("   - pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print("")
            print("3. Use Google Colab (free):")
            print("   - Go to colab.research.google.com")
            print("   - Runtime → Change runtime type → GPU")
            print("-"*50)

        return cuda_available

    except ImportError:
        print("[ERROR] PyTorch not installed!")
        print("        Run: pip install torch")
        return False


def get_model_info(model_name):
    """
    Get model information - explains why we use pretrained models
    """
    print_section("MODEL INFORMATION")

    # Explain pretrained weights
    print("WHY USE PRETRAINED WEIGHTS?")
    print("-"*50)
    print("""
Pretrained models are neural networks that have already been trained on
a large dataset (like COCO with 80 classes, 1.4M+ images). Using them:

1. FASTER TRAINING: Start with learned features instead of random weights
2. BETTER ACCURACY: Leverage knowledge from millions of images
3. LESS DATA NEEDED: Fine-tune with your dataset, not train from scratch
4. STABLE CONVERGGE: Models converge faster with good initial weights

This is called "Transfer Learning" - we transfer knowledge from one task
(COCO object detection) to our task (mosquito detection).
""")

    # Model sizes comparison
    print("\nYOLOv11 MODEL VARIANTS:")
    print("-"*50)
    print(f"  yolo11n  - Nano    (  3.2M params) - Fastest, lowest accuracy")
    print(f"  yolo11s  - Small   (  9.4M params) - Fast, good accuracy")
    print(f"  yolo11m  - Medium  ( 25.9M params) - Balanced")
    print(f"  yolo11l  - Large   ( 53.2M params) - Slow, high accuracy")
    print(f"  yolo11x  - XLarge  ( 97.2M params) - Slowest, highest accuracy")
    print(f"\n→ Using: {model_name} (best for learning/fast experimentation)")
    print(f"\n  Why {model_name}?")
    print(f"  - Smallest model = fastest training")
    print(f"  - Good for learning the workflow")
    print(f"  - Can upgrade to 's', 'm', 'l', 'x' later")
    print(f"  - Lower memory requirements")


def explain_training_parameters():
    """
    Explain the key training parameters
    """
    print_section("TRAINING PARAMETERS EXPLAINED")

    print("""
KEY TRAINING CONCEPTS:
======================

1. EPOCHS (30)
   - Definition: One complete pass through the entire training dataset
   - In our case: 30 passes through 6,633 training images
   - More epochs = more learning, but risk overfitting
   - Baseline: 30 is good starting point. We'll monitor if more are needed.

2. BATCH SIZE (16)
   - Definition: Number of images processed before updating weights
   - Larger batch = more stable gradients, faster training
   - GPU memory limits batch size
   - If you get OOM errors, reduce batch size to 8 or 4

3. IMAGE SIZE (640)
   - Definition: Input image resolution in pixels (width = height)
   - Larger = more detail = better accuracy but slower
   - 640 is standard for YOLO - balances speed and accuracy
   - Your dataset is already 640x640, so this matches perfectly

4. LEARNING RATE (0.01)
   - Definition: How much to adjust weights during each update
   - Too high = unstable training, may diverge
   - Too low = very slow convergence
   - 0.01 is the standard YOLO starting point
   - It decreases automatically during training (scheduler)

5. WEIGHT DECAY (0.0005)
   - Definition: L2 regularization - penalizes large weights
   - Prevents overfitting by encouraging simpler models
   - Small value is standard for YOLO

6. AUGMENTATION (enabled)
   - Definition: Randomly modify images during training
   - Why? Makes model robust to variations
   - Our settings:
     * Horizontal flip: 50% chance (learns mosquitoes facing either direction)
     * No vertical flip (mosquitoes don't fly upside-down)
   - Other augmentations applied automatically by YOLO:
     * Random scaling (0.5-1.5x)
     * Random crop
     * Color jitter (brightness, saturation)
     * Mosaic (combines 4 images)

7. VALIDATION
   - Definition: Evaluating on validation set during training
   - Runs every epoch to monitor performance
   - Uses 670 validation images
   - Tracks: mAP, precision, recall, loss
   - Saves "best" model based on validation mAP
""")


def create_directories():
    """Create necessary directories"""
    print_section("CREATING DIRECTORIES")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Models directory: {MODEL_DIR}")
    print(f"[OK] Results directory: {RESULTS_DIR}")


def train_model():
    """
    Main training function
    """
    print_section("STARTING YOLO TRAINING")

    # Import YOLO
    from ultralytics import YOLO

    # Load the model with pretrained weights
    print(f"\nLoading model: {MODEL_NAME}")
    print(f"Loading pretrained weights from COCO dataset...")

    model = YOLO(f"{MODEL_NAME}.pt")  # Load pretrained yolo11n.pt

    print(f"[OK] Model loaded with pretrained weights")

    # Display model info
    print(f"\nModel: {MODEL_NAME}")
    print(f"Parameters: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")

    # Training configuration
    results = model.train(
        # Data configuration
        data=str(DATA_YAML),

        # Model configuration
        model=MODEL_NAME,
        pretrained=True,

        # Training parameters
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        # Optimizer settings
        lr0=LEARNING_RATE,           # Initial learning rate
        lrf=LEARNING_RATE * 0.01,    # Final learning rate (1% of initial)
        momentum=0.937,              # SGD momentum
        weight_decay=WEIGHT_DECAY,   # L2 regularization

        # Augmentation
        augment=AUGMENT,
        flipud=FLIP_UD,              # Vertical flip
        fliplr=FLIP_LR,              # Horizontal flip

        # Training behavior
        patience=PATIENCE,           # Early stopping
        save=SAVE_BEST,
        save_period=SAVE_PERIOD,
        save_json=SAVE_JSON,

        # Project and names
        project=str(PROJECT_ROOT),
        name="train",                # Training run name
        exist_ok=True,               # Overwrite if exists

        # Other settings
        verbose=True,
        seed=42,                     # Reproducibility
        deterministic=True,          # Deterministic training
    )

    return results


def save_model(results):
    """
    Save the trained model and explain outputs
    """
    print_section("SAVING MODEL AND RESULTS")

    # Find the best model
    train_dir = PROJECT_ROOT / "train"
    weights_dir = train_dir / "weights"

    best_model_path = weights_dir / "best.pt"
    last_model_path = weights_dir / "last.pt"

    # Copy best model to our models directory
    import shutil

    if best_model_path.exists():
        target_path = MODEL_DIR / "best.pt"
        shutil.copy2(best_model_path, target_path)
        print(f"[OK] Best model saved to: {target_path}")
    else:
        print("[WARNING] Best model not found at expected location")

    if last_model_path.exists():
        target_path = MODEL_DIR / "last.pt"
        shutil.copy2(last_model_path, target_path)
        print(f"[OK] Last model saved to: {target_path}")

    # Explain where outputs are saved
    print_section("WHERE YOLO STORES OUTPUTS")

    print("""
YOLO stores all outputs in: {train_dir}
{train_dir}/
├── weights/
│   ├── best.pt         ← Best model (lowest validation loss)
│   └── last.pt         ← Last epoch model
│
├── results.csv         ← Training metrics per epoch
├── results.png         ← Training curves plot
│
├── confusion_matrix.png      ← Confusion matrix (validation)
├── confusion_matrix.csv
│
├── PR_curve.png        ← Precision-Recall curve
├── PR_curve.csv
│
├── labels_correlogram.png    ← Label distribution
│
├── predictions.json   ← Validation predictions
│
└── (many more files...)
""".format(train_dir=train_dir))

    # List all output files
    print("\nFiles generated:")
    for f in sorted(train_dir.glob("*")):
        if f.is_file():
            print(f"  - {f.name}")

    print(f"\nResults directory: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*")):
        if f.is_file():
            print(f"  - {f.name}")


def print_training_summary():
    """Print training completion summary"""
    print_section("TRAINING COMPLETE!")

    print("""
SUCCESS! Your mosquito detection model has been trained.

NEXT STEPS:
-----------
1. Check the results:
   - Look at results.png for training curves
   - Check confusion_matrix.png to see predictions
   - Review PR_curve.png for precision/recall

2. Test the model:
   python predict.py --image path/to/image.jpg

3. Evaluate on test set:
   python evaluate.py

4. If accuracy is low, try:
   - Increase epochs (50-100)
   - Use larger model (yolo11s, yolo11m)
   - Tune hyperparameters
   - Add more data

MODEL LOCATION:
---------------
   models/best.pt  ← Use this for inference
""")

    print(f"\nBest model: {MODEL_DIR / 'best.pt'}")
    print(f"Training results: {PROJECT_ROOT / 'train'}")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("  🦟 MOSQUITO DETECTION - YOLO TRAINING")
    print("  Stage 4: Train First Baseline Model")
    print("="*70)

    # Step 1: Check version
    version = check_ultralytics_version()

    # Step 2: Check GPU
    cuda_available = check_gpu_available()

    # Step 3: Show model info
    get_model_info(MODEL_NAME)

    # Step 4: Explain parameters
    explain_training_parameters()

    # Step 5: Create directories
    create_directories()

    # Step 6: Check data.yaml
    print_section("VERIFYING DATA CONFIGURATION")
    if DATA_YAML.exists():
        print(f"[OK] Found: {DATA_YAML}")
    else:
        print(f"[ERROR] data.yaml not found: {DATA_YAML}")
        print("        Please run Stage 3 setup first!")
        sys.exit(1)

    # Confirm training
    print("\n" + "="*50)
    print("READY TO TRAIN")
    print("="*50)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Device: {'GPU' if cuda_available else 'CPU'}")
    print("="*50)

    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return

    # Step 7: Train
    print("\n" + "="*50)
    print("TRAINING IN PROGRESS...")
    print("="*50)
    print("This may take several minutes to hours depending on hardware.")
    print("Press Ctrl+C to stop (but you won't save the model).\n")

    try:
        results = train_model()

        # Step 8: Save and summarize
        save_model(results)
        print_training_summary()

    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
        print("No model saved. Run again to complete training.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()