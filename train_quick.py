"""
Quick Training Script - No prompts
====================================
Run this for automated training
"""

import sys
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_YAML = PROJECT_ROOT / "data.yaml"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Training parameters
MODEL_NAME = "yolo11n"
IMAGE_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = "auto"  # Auto-detect GPU/CPU


def main():
    print("="*70)
    print("MOSQUITO DETECTION - YOLO TRAINING (Quick Start)")
    print("="*70)

    # Check Ultralytics
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("ERROR: Ultralytics not installed")
        print("Run: pip install ultralytics")
        sys.exit(1)

    # Check GPU
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA available: Yes")
        else:
            print("CUDA available: No (using CPU)")
            print("NOTE: Training on CPU is slow. Consider Kaggle for GPU.")
    except ImportError:
        print("PyTorch not installed")

    # Check data.yaml
    if not DATA_YAML.exists():
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        print("Run setup_project_structure.py first")
        sys.exit(1)

    print(f"\nStarting training...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Device: {DEVICE}")

    # Import and train
    from ultralytics import YOLO

    # Load pretrained model
    model = YOLO(f"{MODEL_NAME}.pt")

    # Train
    results = model.train(
        data=str(DATA_YAML),
        model=MODEL_NAME,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(PROJECT_ROOT),
        name="train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
    )

    # Copy best model
    import shutil
    train_dir = PROJECT_ROOT / "train"
    best_weights = train_dir / "weights" / "best.pt"

    if best_weights.exists():
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        shutil.copy2(best_weights, MODEL_DIR / "best.pt")
        print(f"\n[OK] Best model saved to: {MODEL_DIR / 'best.pt'}")
    else:
        print("\n[WARNING] Best model not found")

    print("\nTraining complete!")
    print(f"Results: {train_dir}")


if __name__ == "__main__":
    main()