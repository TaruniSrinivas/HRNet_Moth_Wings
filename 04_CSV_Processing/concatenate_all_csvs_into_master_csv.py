# ==============================================================================
# SCRIPT PURPOSE: Scans an input directory for multiple individual CSV annotation
#                 files and concatenates them into a single consolidated master CSV file.
# INPUTS:         --input  : Directory path containing individual CSV files (default: ./data/raw)
# OUTPUTS:        --output : Output file path for the merged CSV (default: ./data/processed/combined.csv)
# PAPER SECTION:  Methods - Section 2.1 (Data Ingestion & Dataset Consolidation)
# ==============================================================================

import argparse
import glob
import os
import pandas as pd


def append_csv_files(input_dir, output_file):
    # Resolve absolute paths
    input_dir = os.path.abspath(input_dir)
    output_file = os.path.abspath(output_file)

    print(f"Scanning directory: {input_dir}")

    # Use glob to find all .csv files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # FIX 1: Exclude the output file if it sits inside the input directory
    csv_files = [
        f for f in csv_files if os.path.abspath(f) != output_file
    ]

    if not csv_files:
        print("Error: No valid CSV files found to process.")
        return

    print(f"Found {len(csv_files)} CSV files. Beginning extraction...\n")

    df_list = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(
                f"Successfully loaded: {os.path.basename(file)} ({len(df)} rows)"
            )
        except Exception as e:
            print(f"Failed to read {os.path.basename(file)}: {e}")

    # FIX 2: Check if df_list is empty before concatenating
    if not df_list:
        print("Error: No dataframes were successfully loaded.")
        return

    print("\nAppending files together...")
    master_df = pd.concat(df_list, ignore_index=True)

    # Ensure target output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    master_df.to_csv(output_file, index=False)

    print(f"\nSuccess! All files appended.")
    print(f"Master file saved to: {output_file}")
    print(f"Total rows in final dataset: {len(master_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine CSV files into a single master file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/raw",
        help="Path to input directory containing CSVs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/combined.csv",
        help="Path to output CSV file",
    )

    args = parser.parse_args()
    append_csv_files(args.input, args.output)
