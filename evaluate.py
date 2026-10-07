"""
YOLO Mosquito Detection - Evaluation Script
===========================================
Stage 6: Evaluate the trained model on the held-out test set

This script:
1. Loads models/best.pt
2. Evaluates on the test set
3. Generates and prints standard YOLO metrics
4. Saves evaluation results and plots
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DATA_YAML = PROJECT_ROOT / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation" # For evaluation specific outputs
# =============================================================================

def load_class_names(data_yaml_path):
    """Load class names from data.yaml"""
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get('names', [])

def evaluate_model():
    """
    Load the model, run evaluation on the test set, and print results.
    """
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Please train a model first (Stage 4).")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found: {DATA_YAML}")
        print("Please ensure your project is set up correctly (Stage 3).")
        sys.exit(1)

    # Load model
    print(f"\n[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Create output directory for evaluation results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation on the test set
    # The 'split="test"' argument specifically uses the test data defined in data.yaml
    print(f"[INFO] Running evaluation on test set using {DATA_YAML}...")
    
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640, # Ensure this matches training image size
        conf=0.25, # Confidence threshold for NMS
        iou=0.7,   # IoU threshold for NMS
        project=str(OUTPUT_DIR.parent), # Save output to results/ directory
        name="evaluation", # Subdirectory for this run (results/evaluation/)
        exist_ok=True, # Overwrite if exists
        save_json=True, # Save metrics to a JSON file
        save_hybrid=True, # Save labels+predictions for inspection
    )

    # Print overall metrics
    print("\n" + "="*70)
    print("        MODEL EVALUATION RESULTS (OVERALL)        ")
    print("="*70)
    print(f"  mAP@50 (Mean Average Precision at IoU=0.50):    {metrics.results_dict['metrics/mAP50(B)'][0]:.4f}")
    print(f"  mAP@50-95 (Mean Average Precision @ IoU=0.50-0.95): {metrics.results_dict['metrics/mAP50-95(B)'][0]:.4f}")
    print(f"  Precision:                                    {metrics.results_dict['metrics/precision(B)'][0]:.4f}")
    print(f"  Recall:                                       {metrics.results_dict['metrics/recall(B)'][0]:.4f}")
    print("="*70)

    # Print per-class metrics
    class_names = load_class_names(DATA_YAML)
    print("\n" + "="*70)
    print("        MODEL EVALUATION RESULTS (PER CLASS)       ")
    print("="*70)
    for i, name in enumerate(class_names):
        try:
            ap_class = metrics.ap_per_class[i]
            print(f"  Class {i:2d} ({name:20s}): AP={ap_class[0]:.4f}  Precision={ap_class[1]:.4f}  Recall={ap_class[2]:.4f}")
        except IndexError:
            print(f"  Class {i:2d} ({name:20s}): No detections or metrics available.")
    print("="*70)

    # Explain where results are saved
    print("\n[INFO] Evaluation results saved to:")
    eval_run_dir = OUTPUT_DIR / "evaluation" # Ultralytics creates a new run folder like 'evaluation'
    print(f"  - Plots (confusion_matrix.png, P_curve.png, R_curve.png): {eval_run_dir}")
    print(f"  - Metrics JSON: {eval_run_dir}/results.json")
    print(f"  - Predictions (labels.json): {eval_run_dir}/labels.json and {eval_run_dir}/predictions.json")
    print(f"  - Sample prediction images: {eval_run_dir}/val_batchX_pred.jpg (for debugging)")

def main():
    print("\n" + "="*70)
    print("  🦟 MOSQUITO DETECTION - MODEL EVALUATION")
    print("  Stage 6: Evaluate Baseline Model")
    print("="*70)
    evaluate_model()

if __name__ == "__main__":
    main()
