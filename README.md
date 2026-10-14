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

## Execution Guide

To reproduce the analysis, execute the scripts in numerical order according to your data preparation and model pipeline needs:

### 1. Environment Setup
```bash
conda env create -f 01_Setup/environment_set-up.yml
conda activate hrnet_env

# Clean metadata tags and extract scale values
python 02_TPS_Processing/clean_tps_metadata_tags.py
python 02_TPS_Processing/extract_scale_values_from_tps.py

# Optional helper scripts for folder organization & cleaning
bash 02_TPS_Processing/flatten_nested_image_folders.sh
bash 02_TPS_Processing/extract_id_patterns_from_filenames.sh
bash 02_TPS_Processing/remove_ventral_images.sh
bash 02_TPS_Processing/filter_uncommon_files_between_datasets.sh

#IImage Pre-processing
python 03_Image_Processing/crop_images_from_landmark_bboxes.py

# Concatenate and split dataset annotations
python 04_CSV_Processing/concatenate_all_csvs_into_master_csv.py
python 04_CSV_Processing/split_master_csv_into_train_val_test.py

# Convert CSV annotations to COCO JSON format
python 04_CSV_Processing/convert_single_csv_to_coco_json.py
python 04_CSV_Processing/convert_batch_csvs_to_coco_jsons.py

# Train HRNet model
python 05_ML_Models/train_hrnet.py --cfg 01_Setup/hrnet_train.yaml

# Run landmark inference
python 05_ML_Models/predict_hrnet.py --cfg 01_Setup/hrnet_predict.yaml

# Convert predicted landmark coordinates back to TPS format
python 06_Post_Processing/convert_predicted_landmarks_csv_to_tps.py

# Generate comparison plots (Machine vs. Human landmarks)
python 06_Post_Processing/overlay_machine_vs_human_landmarks.py
