# ==============================================================================
# SCRIPT PURPOSE: Performs a genus-level stratified train/validation/test split on a
#                 master CSV dataset to prevent data leakage across taxonomy classes.
# INPUTS:         --input      : Path to the master CSV file (default: ./data/master.csv)
#                 --train_pct  : Percentage for training split (default: 70.0)
#                 --val_pct    : Percentage for validation split (default: 15.0)
#                 --test_pct   : Percentage for test split (default: 15.0)
# OUTPUTS:        --output_dir : Directory where train_data.csv, val_data.csv, and test_data.csv are saved
# PAPER SECTION:  Methods - Section 2.2 (Dataset Stratification & Splitting)
# ==============================================================================

import argparse
import math
import os
import numpy as np
import pandas as pd


def extract_genus(filename):
    """Extracts the first word (Genus) from the filename string."""
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split(" ")
        return parts[0] if len(parts) > 0 else "Unknown"
    except Exception:
        return "Unknown"


def split_dataset(input_csv, output_dir, train_pct, val_pct, test_pct, seed=42):
    # Validate split ratios
    total_pct = train_pct + val_pct + test_pct
    if round(total_pct, 2) != 100.0:
        raise ValueError(f"Percentages must sum to 100. Got {total_pct}")

    train_ratio = train_pct / 100.0
    val_ratio = val_pct / 100.0

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    print(f"Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    # Extract Genus metadata per unique image
    image_metadata = (
        df[["image_id", "file_name"]].drop_duplicates().copy()
    )
    image_metadata["genus"] = image_metadata["file_name"].apply(extract_genus)

    train_ids, val_ids, test_ids = [], [], []
    np.random.seed(seed)

    grouped_images = image_metadata.groupby("genus")

    print("\n--- Stratification Breakdown ---")
    for genus, group in grouped_images:
        images = group["image_id"].tolist()
        np.random.shuffle(images)
        n = len(images)

        # Handle rare class edge cases
        if n == 1:
            train_ids.extend(images)
            print(f"{genus}: 1 total -> [Train: 1, Val: 0, Test: 0]")
        elif n == 2:
            train_ids.append(images[0])
            test_ids.append(images[1])
            print(f"{genus}: 2 total -> [Train: 1, Val: 0, Test: 1]")
        else:
            t_end = math.ceil(n * train_ratio)
            v_end = t_end + max(1, int(n * val_ratio))
            if v_end > n:
                v_end = n

            t_slice = images[:t_end]
            v_slice = images[t_end:v_end]
            test_slice = images[v_end:]

            train_ids.extend(t_slice)
            val_ids.extend(v_slice)
            test_ids.extend(test_slice)
            print(
                f"{genus}: {n} total -> [Train: {len(t_slice)}, Val: {len(v_slice)}, Test: {len(test_slice)}]"
            )

    # Create subset DataFrames
    train_df = df[df["image_id"].isin(train_ids)].copy()
    val_df = df[df["image_id"].isin(val_ids)].copy()
    test_df = df[df["image_id"].isin(test_ids)].copy()

    # Integrity Check
    overlap = (
        set(train_ids).intersection(set(val_ids))
        | set(train_ids).intersection(set(test_ids))
        | set(val_ids).intersection(set(test_ids))
    )
    if overlap:
        print("\nCRITICAL WARNING: Image IDs overlap between sets!")
    else:
        print("\nIntegrity Check Passed: No overlapping images across splits.")

    # Save output CSVs
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_data.csv")
    val_path = os.path.join(output_dir, "val_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n--- Final Summary ---")
    print(
        f"Train: {len(train_ids)} images ({len(train_df)} rows) -> Saved to {train_path}"
    )
    print(
        f"Val:   {len(val_ids)} images ({len(val_df)} rows) -> Saved to {val_path}"
    )
    print(
        f"Test:  {len(test_ids)} images ({len(test_df)} rows) -> Saved to {test_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratify-split CSV dataset into train, val, and test sets."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/master.csv",
        help="Path to master CSV dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/splits",
        help="Directory to save split CSVs",
    )
    parser.add_argument(
        "--train_pct",
        type=float,
        default=70.0,
        help="Train split percentage (default: 70)",
    )
    parser.add_argument(
        "--val_pct",
        type=float,
        default=15.0,
        help="Validation split percentage (default: 15)",
    )
    parser.add_argument(
        "--test_pct",
        type=float,
        default=15.0,
        help="Test split percentage (default: 15)",
    )

    args = parser.parse_args()
    split_dataset(
        args.input,
        args.output_dir,
        args.train_pct,
        args.val_pct,
        args.test_pct,
    )
