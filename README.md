# HRNet_Moth_Wings

This repository contains the dataset processing, format conversion, HRNet model training, inference, and post-processing scripts used for landmark detection on moth forewings and hindwings.

---

## Repository Structure

```text
.
├── 01_CSV_Operations/         # Dataset merging, splitting, and COCO JSON conversion
├── 02_Image_Processing/       # Specimen image cropping based on landmark bounding boxes
├── 03_ML_Models/              # HRNet model architecture, dataset class, training, and inference
├── 04_Post_Processing/        # Exporting predictions to TPS format and generating overlay visual checks
├── 05_Setup/                  # Environment setup file (.yml) and HRNet configuration files (.yaml)
└── 06_TPS_Processing/         # TPS metadata cleaning, scale extraction, and dataset pruning
