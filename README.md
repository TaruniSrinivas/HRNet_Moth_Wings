# Environment Setup
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

# Image Pre-processing
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
