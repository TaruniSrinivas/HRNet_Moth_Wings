# ==============================================================================
# SCRIPT PURPOSE: Converts a single CSV file containing 12 landmark keypoint annotations,
#                 bounding boxes, and image metadata into standard COCO JSON format.
# INPUTS:         --input         : Path to the input CSV file
#                 --category_name : Category label (default: "Hindwing")
#                 --category_id   : Numeric category ID (default: 1)
# OUTPUTS:        --output        : Path where the resulting COCO JSON file will be written
# PAPER SECTION:  Methods - Section 2.3 (Landmark Schema & COCO Conversion)
# ==============================================================================

import argparse
from datetime import datetime
import json
import os
import pandas as pd


def convert_csv_to_coco(
    csv_file_path,
    output_json_path,
    category_name="Hindwing",
    category_id=1,
    supercategory="Moth",
):
    """Converts a CSV file with 12 landmark keypoint annotations into COCO JSON format."""
    NUM_KEYPOINTS = 12  # Hardcoded specifically for the 12 landmark schema

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file_path}': {e}")
        return

    coco_data = {
        "info": {
            "description": f"Custom Keypoint Dataset for {category_name} converted from CSV to COCO format",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": supercategory,
                "keypoints": [
                    f"keypoint_{i}" for i in range(1, NUM_KEYPOINTS + 1)
                ],
                "skeleton": [],
            }
        ],
    }

    # Extract unique image records
    unique_images = df[
        ["image_id", "file_name", "image_height", "image_width"]
    ].drop_duplicates()

    for _, row in unique_images.iterrows():
        coco_data["images"].append(
            {
                "id": int(row["image_id"]),
                "file_name": str(row["file_name"]),
                "height": int(row["image_height"]),
                "width": int(row["image_width"]),
            }
        )

    # Extract 12 keypoints per annotation row
    for _, row in df.iterrows():
        keypoints = []
        for i in range(1, NUM_KEYPOINTS + 1):
            keypoints.append(float(row[f"keypoint_x_{i}"]))
            keypoints.append(float(row[f"keypoint_y_{i}"]))
            keypoints.append(int(row[f"keypoint_visibility_{i}"]))

        coco_data["annotations"].append(
            {
                "image_id": int(row["image_id"]),
                "id": int(row["annotation_id"]),
                "keypoints": keypoints,
                "num_keypoints": int(
                    row.get("num_keypoints_total", NUM_KEYPOINTS)
                ),
                "bbox": [
                    float(row["bbox_x_tl"]),
                    float(row["bbox_y_tl"]),
                    float(row["bbox_width"]),
                    float(row["bbox_height"]),
                ],
                "iscrowd": int(row.get("iscrowd", 0)),
                "area": float(row["area"]),
                "category_id": category_id,  # Guaranteed match with categories block
            }
        )

    try:
        with open(output_json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        print(f"Conversion complete! COCO JSON saved to: {output_json_path}")
    except IOError as e:
        print(f"Error writing JSON file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 12-landmark CSV annotations to COCO JSON format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/Master_Hindwing_Geometrids.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/Master_Hindwing_Geometrids.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--category_name",
        type=str,
        default="Hindwing",
        help="Category name (default: Hindwing)",
    )
    parser.add_argument(
        "--category_id",
        type=int,
        default=1,
        help="Category ID (default: 1)",
    )

    args = parser.parse_args()
    convert_csv_to_coco(
        csv_file_path=args.input,
        output_json_path=args.output,
        category_name=args.category_name,
        category_id=args.category_id,
    )
