"""
YOLO Mosquito Detection - Inference Script
==========================================
Stage 5: Detect mosquitoes in static images or folders

This script:
1. Loads models/best.pt
2. Runs inference on input image(s)
3. Draws bounding boxes and labels
4. Prints detection details to terminal
5. Saves results to results/predictions/
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "predictions"
# =============================================================================

def predict(source, conf_threshold=0.25):
    """
    Run inference on a single image or folder.
    """
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Please train a model first (Stage 4).")
        return

    # Load model
    model = YOLO(str(MODEL_PATH))
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Running inference on: {source}")
    print(f"[INFO] Confidence threshold: {conf_threshold}")
    
    # Run inference
    # conf: Confidence threshold (filters out low-confidence detections)
    results = model.predict(source=source, conf=conf_threshold, save=False, verbose=False)

    if not isinstance(results, list):
        results = [results]

    for i, result in enumerate(results):
        # Save visualization
        output_path = OUTPUT_DIR / f"prediction_{i}.jpg"
        result.save(filename=str(output_path))
        
        # Print details to terminal
        print(f"\nResults for image: {result.path}")
        
        boxes = result.boxes
        if len(boxes) == 0:
            print("  No mosquitoes detected.")
            continue
            
        for box in boxes:
            # Get class name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Get coordinates (x1, y1, x2, y2)
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(x) for x in coords]
            
            print(f"  - Class: {class_name:20s} | Conf: {conf:.2f} | Coords: ({x1}, {y1}, {x2}, {y2})")
            
        print(f"  Saved visualization to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Prediction Script")
    parser.add_argument("--source", type=str, required=True, 
                        help="Path to image or folder of images")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()
    
    predict(args.source, args.conf)