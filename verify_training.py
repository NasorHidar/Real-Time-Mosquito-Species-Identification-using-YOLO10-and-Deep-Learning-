"""
Training Verification Script
=============================
Verify that training completed successfully
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def verify_training():
    """Check all expected outputs from training"""
    print("\n" + "="*70)
    print("TRAINING VERIFICATION")
    print("="*70)

    checks = {}

    # Check 1: Best model exists
    best_model = PROJECT_ROOT / "models" / "best.pt"
    checks["Best model (models/best.pt)"] = best_model.exists()

    # Check 2: Last model exists
    last_model = PROJECT_ROOT / "models" / "last.pt"
    checks["Last model (models/last.pt)"] = last_model.exists()

    # Check 3: Training results directory
    train_dir = PROJECT_ROOT / "train"
    checks["Training directory exists"] = train_dir.exists()

    # Check 4: Weights in train directory
    weights_dir = train_dir / "weights"
    checks["Weights directory exists"] = weights_dir.exists()

    # Check 5: Results CSV
    results_csv = train_dir / "results.csv"
    checks["Results CSV (training metrics)"] = results_csv.exists()

    # Check 6: Training curves
    results_png = train_dir / "results.png"
    checks["Training curves (results.png)"] = results_png.exists()

    # Check 7: Confusion matrix
    confusion_matrix = train_dir / "confusion_matrix.png"
    checks["Confusion matrix"] = confusion_matrix.exists()

    # Check 8: PR curve
    pr_curve = train_dir / "PR_curve.png"
    checks["PR curve"] = pr_curve.exists()

    # Check 9: Model file size (should be > 0)
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 * 1024)
        checks[f"Best model size ({size_mb:.1f} MB)"] = size_mb > 1
    else:
        checks["Best model size check"] = False

    # Print results
    print("\nVerification Results:")
    print("-"*70)

    all_passed = True
    for check_name, passed in checks.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    # Print summary
    print("\n" + "="*70)

    if all_passed:
        print("ALL CHECKS PASSED - Training completed successfully!")
        print("\nYour trained model is ready:")
        print(f"  Location: {best_model}")
        print(f"  Size: {best_model.stat().st_size / (1024*1024):.1f} MB")
        print("\nNext steps:")
        print("  1. View training results: train/results.png")
        print("  2. View confusion matrix: train/confusion_matrix.png")
        print("  3. Run predictions: python predict.py")
        print("  4. Evaluate on test set: python evaluate.py")
    else:
        print("SOME CHECKS FAILED - Training may not have completed")
        print("\nTroubleshooting:")
        print("  1. Check for error messages during training")
        print("  2. Ensure training ran for full epochs")
        print("  3. Check disk space")
        print("  4. Verify data.yaml is correct")

    print("="*70)

    return all_passed


def print_training_outputs_guide():
    """Print guide to all training outputs"""
    guide = """
TRAINING OUTPUTS GUIDE
======================

After training, YOLO creates these files in: mosquito-detection/train/

1. WEIGHTS (most important)
   └── weights/
       ├── best.pt      ← Best model (lowest validation loss) - USE THIS
       └── last.pt      ← Model from last epoch

2. TRAINING CURVES
   └── results.png      ← Shows loss and metrics over epochs
       - Train loss (should decrease)
       - Val loss (should decrease)
       - mAP@50 (should increase)
       - mAP@50-95 (should increase)

3. CONFUSION MATRIX
   └── confusion_matrix.png
       - Shows which classes get confused
       - Diagonal = correct predictions
       - Off-diagonal = misclassifications

4. PRECISION-RECALL CURVE
   └── PR_curve.png
       - Shows precision vs recall trade-off
       - Higher area under curve = better

5. METRICS CSV
   └── results.csv
       - All metrics for every epoch
       - Can be loaded in Excel/Python for analysis

6. VALIDATION PREDICTIONS
   └── predictions.json
       - Raw predictions on validation set
       - Bounding boxes, confidence scores

7. LABEL DISTRIBUTION
   └── labels_correlogram.png
       - Shows distribution of bounding boxes
       - Center positions, sizes

INTERPRETING RESULTS:
---------------------

GOOD TRAINING:
  ✓ Train loss decreases smoothly
  ✓ Val loss decreases (or stays stable)
  ✓ mAP@50 > 0.7 (70%+)
  ✓ Confusion matrix has strong diagonal
  ✓ PR curve has high area

OVERFITTING:
  ✗ Train loss keeps decreasing
  ✗ Val loss starts increasing
  → Solution: Reduce epochs, add augmentation, use smaller model

UNDERFITTING:
  ✗ Both losses stay high
  ✗ mAP stays low
  → Solution: Train longer, use larger model, check data quality

CLASS IMBALANCE:
  ✗ Some classes have many more samples
  ✗ Model biased toward majority class
  → Solution: Balance dataset, use class weights
    """
    print(guide)


def main():
    """Main verification"""
    import argparse

    parser = argparse.ArgumentParser(description="Verify training completed")
    parser.add_argument("--guide", action="store_true",
                       help="Show guide to training outputs")
    args = parser.parse_args()

    if args.guide:
        print_training_outputs_guide()
    else:
        success = verify_training()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()