# ==============================================================================
# SCRIPT PURPOSE: Reads a landmark TPS file and uses regular expressions to extract
#                 all embedded scale conversion factors ('SCALE=' values).
# INPUTS:         --input : Path to input TPS file (default: ./data/sample.TPS)
# OUTPUTS:        Prints list of float scale conversion values parsed from the file
# PAPER SECTION:  Methods - Section 2.5 (Morphometric Scale Metadata Extraction)
# ==============================================================================

import argparse
import os
import re


def get_scales_from_tps_file(file_path):
    """Extracts all 'SCALE=' values from a single TPS file.

    Args:
        file_path (str): Full path to the TPS file.

    Returns:
        list: List of float scale values found in the file.
    """
    scales = []

    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return []

    if not os.path.isfile(file_path):
        print(f"Error: Path '{file_path}' is not a file.")
        return []

    # Enhanced regex to capture integers, decimals, and scientific notation
    scale_pattern = re.compile(
        r"SCALE\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", re.IGNORECASE
    )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                match = scale_pattern.search(line)
                if match:
                    try:
                        scale_value = float(match.group(1))
                        scales.append(scale_value)
                    except ValueError:
                        print(
                            f"Warning: Could not parse scale value on line {line_num}: '{line.strip()}' in '{file_path}'."
                        )

    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []

    return scales


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SCALE values from a landmark TPS file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/sample.TPS",
        help="Path to input TPS file (default: ./data/sample.TPS)",
    )

    args = parser.parse_args()
    found_scales = get_scales_from_tps_file(args.input)

    if found_scales:
        print(f"\nScales found in '{args.input}':")
        for idx, scale in enumerate(found_scales, 1):
            print(f"  [{idx}] {scale}")
    else:
        print(f"\nNo scales found in '{args.input}'.")
