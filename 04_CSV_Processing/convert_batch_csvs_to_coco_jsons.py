# ==============================================================================
# SCRIPT PURPOSE: Batch converts multiple CSV annotation files into COCO JSON files.
#                 Automatically determines and maps wing categories (Forewing, Hindwing,
#                 Moth_Wing) and IDs based on keywords in the CSV filenames.
# INPUTS:         --input  : Directory containing raw CSV annotation files (default: ./data/csv_raw)
# OUTPUTS:        --output : Directory where converted COCO JSON files will be saved (default: ./data/json_coco)
# PAPER SECTION:  Methods - Section 2.3 (Dataset Formatting for Model Training)
# ==============================================================================

import argparse
import glob
import os
from csvtoCOCO import convert_csv_to_coco

# Mapping category names to distinct COCO category IDs
CATEGORY_MAP = {"Forewing": 1, "Hindwing": 2, "Moth_Wing": 3}


def process_all_csvs(input_dir, output_dir):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return 0, 0

    print(f"Found {len(csv_files)} CSV files. Starting conversion...\n")

    success_count = 0
    failure_count = 0

    for csv_path in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}.json")

        # Category determination logic
        name_lower = base_name.lower()
        if "forewing" in name_lower:
            wing_category = "Forewing"
        elif "hindwing" in name_lower:
            wing_category = "Hindwing"
        else:
            wing_category = "Moth_Wing"

        category_id = CATEGORY_MAP.get(wing_category, 3)

        print(
            f"Converting: {base_name}.csv -> Tagged as '{wing_category}' (ID: {category_id})"
        )

        # Fault-tolerant call to converter
        try:
            convert_csv_to_coco(
                csv_file_path=csv_path,
                output_json_path=json_path,
                category_name=wing_category,
                category_id=category_id,
            )
            success_count += 1
        except Exception as e:
            print(f"Failed to convert {base_name}.csv: {e}")
            failure_count += 1

    print(f"\nBatch processing complete!")
    print(f"Successfully converted: {success_count} files")
    if failure_count > 0:
        print(f"Failed conversions: {failure_count} files")

    return success_count, failure_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch convert CSV keypoint annotations to COCO JSON format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/csv_raw",
        help="Path to input directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/json_coco",
        help="Path to directory where COCO JSON files will be saved",
    )

    args = parser.parse_args()
    process_all_csvs(args.input, args.output)
