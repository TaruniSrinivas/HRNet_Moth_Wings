# HRNet_Moth_Wings

This repository contains the setup configurations, TPS metadata handling, image processing, CSV format conversion, HRNet model training, inference, and post-processing scripts used for landmark detection on moth forewings and hindwings.

---

## Repository Structure

```text
.
├── 01_Setup/                  # Environment configuration (.yml) and HRNet configuration files (.yaml)
├── 02_TPS_Processing/         # TPS metadata cleaning, scale extraction, and dataset pruning
├── 03_Image_Processing/       # Specimen image cropping based on landmark bounding boxes
├── 04_CSV_Operations/         # Dataset merging, splitting, and COCO JSON conversion
├── 05_ML_Models/              # HRNet model architecture, dataset class, training, and inference
└── 06_Post_Processing/        # Exporting predictions to TPS format and generating overlay visual checks
