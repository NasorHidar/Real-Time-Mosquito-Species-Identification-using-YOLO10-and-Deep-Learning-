# Mosquito Detection Experiment Log

## Project Overview
- **Goal**: Build a mosquito detection system using YOLO
- **Dataset**: Mosquito Detection Dataset v4 (Roboflow)
- **Classes**: 6 mosquito species (aegypti, albopictus, anopheles, culex, culiseta, japonicus-koreicus)

---

## Stage 1: Dataset Verification (2026-09-30)

### Dataset Details
- **Source**: Roboflow Universe
- **Format**: YOLOv11 txt format
- **Total Images**: 7,672
- **Image Size**: 640x640 pixels
- **License**: CC BY 4.0

### Split Distribution
| Split | Images | Labels |
|-------|--------|--------|
| Train | 6,633 | 6,633 |
| Valid | 670 | 670 |
| Test | 369 | 369 |

### Verification Results
- [x] Folder structure correct
- [x] data.yaml valid
- [x] Image-label pairs match
- [x] Label format valid (YOLO txt)
- [x] Images not corrupted

### Observations
- Dataset is already preprocessed (resized to 640x640)
- No augmentation applied in this version
- Ready for YOLO training without modification

---

## Stage 2: Inspect Images and Annotations
*(To be completed)*

---

## Stage 3: Understand YOLO Annotation Format
*(To be completed)*

---

## Stage 4: Prepare Dataset and data.yaml
*(To be completed)*

---

## Stage 5: Train First YOLO Model
*(To be completed)*

