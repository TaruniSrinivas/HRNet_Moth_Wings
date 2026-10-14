# ==============================================================================
# SCRIPT PURPOSE: Converts predicted 12-landmark CSV coordinate files into TPS format,
#                 preserving image reference names and looking up SCALE metadata.
# INPUTS:         --csv        : CSV file containing landmark prediction coordinates
#                 --master_tps : (Optional) Reference TPS file to extract SCALE metadata
# OUTPUTS:        --output_tps : File path where output .TPS file will be saved
# PAPER SECTION:  Methods - Section 2.5 (Post-Processing & Morphometric TPS Export)
# ==============================================================================

import argparse
import os
import pandas as pd


def convert_csv_to_tps(csv_file, master_tps_file, output_tps_file, num_landmarks=12):
    """Converts landmark coordinate CSV predictions into a TPS file with scale metadata lookup."""
    if not os.path.exists(csv_file):
        print(f"Error: Input CSV file not found at '{csv_file}'")
        return False

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        return False

    # Extract SCALE metadata lookup table from master TPS file if available
    scale_lookup = {}
    if master_tps_file and os.path.exists(master_tps_file):
        current_image = None
        with open(master_tps_file, "r") as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith("IMAGE="):
                    current_image = line_str.split("=")[1]
                elif line_str.startswith("SCALE=") and current_image:
                    scale_lookup[current_image] = line_str
    elif master_tps_file:
        print(f"Warning: Master TPS file '{master_tps_file}' not found. Falling back to default SCALE=1.0000.")

    os.makedirs(os.path.dirname(os.path.abspath(output_tps_file)), exist_ok=True)

    with open(output_tps_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f"LM={num_landmarks}\n")

            # Write landmark coordinates
            for i in range(1, num_landmarks + 1):
                x_val = row.get(f"X{i}", 0.0)
                y_val = row.get(f"Y{i}", 0.0)
                f.write(f"{x_val:.5f} {y_val:.5f}\n")

            # Write specimen filename and scale metadata
            img_name = row.get("File_name", "unknown")
            f.write(f"IMAGE={img_name}\n")
            scale_line = scale_lookup.get(img_name, "SCALE=1.0000")
            f.write(f"{scale_line}\n\n")

    print(f"Successfully generated TPS file: '{output_tps_file}'")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert landmark coordinate CSV predictions into TPS format.")
    parser.add_argument(
        "--csv",
        type=str,
        default="./data/predictions.csv",
        help="Path to input coordinate CSV file",
    )
    parser.add_argument(
        "--master_tps",
        type=str,
        default=None,
        help="Optional path to master TPS file for SCALE metadata lookup",
    )
    parser.add_argument(
        "--output_tps",
        type=str,
        default="./data/output_landmarks.TPS",
        help="Path to output TPS file",
    )
    parser.add_argument(
        "--num_landmarks",
        type=int,
        default=12,
        help="Number of landmarks per specimen (default: 12)",
    )

    args = parser.parse_args()
    convert_csv_to_tps(
        csv_file=args.csv,
        master_tps_file=args.master_tps,
        output_tps_file=args.output_tps,
        num_landmarks=args.num_landmarks,
    )
