# ==============================================================================
# SCRIPT PURPOSE: Scans landmark TPS files to verify required metadata tags ('SCALE=' 
#                 and 'LM=12') and cleans unnecessary standalone 'ID=0' tags.
# INPUTS:         --dir      : Path to directory containing TPS files (default: ./data/tps)
#                 --suffix   : File suffix filter ('h' for hindwing, 'f' for forewing)
#                 --dry_run  : Optional flag to preview changes without modifying files
# OUTPUTS:        Cleaned TPS files with removed 'ID=0' tags and printed validation summary
# PAPER SECTION:  Methods - Section 2.5 (Morphometric TPS Metadata Cleaning)
# ==============================================================================

import argparse
import os


def analyze_and_modify_tps(directory_path, tps_suffix_char, dry_run=False):
    """Analyzes TPS files for required metadata ('SCALE=' and 'LM=12')

    and removes standalone 'ID=0' tags cleanly.
    """
    directory_path = os.path.abspath(directory_path)
    print(
        f"--- Analyzing TPS files in: {directory_path} (Suffix: *{tps_suffix_char}.tps) ---"
    )

    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return None

    # Filter TPS files matching suffix
    suffix = f"{tps_suffix_char}.tps".lower()
    tps_files = [
        f for f in os.listdir(directory_path) if f.lower().endswith(suffix)
    ]

    if not tps_files:
        print("No matching TPS files found.")
        return {
            "total": 0,
            "missing_scale": [],
            "missing_lm12": [],
            "modified": [],
        }

    files_without_scale = []
    files_without_lm12 = []
    modified_files = []

    for tps_file_name in tps_files:
        tps_file_path = os.path.join(directory_path, tps_file_name)

        try:
            with open(tps_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            full_text = "".join(lines)

            if "SCALE=" not in full_text:
                files_without_scale.append(tps_file_name)
            if "LM=12" not in full_text:
                files_without_lm12.append(tps_file_name)

            # Cleanly remove lines that only contain "ID=0"
            new_lines = [line for line in lines if line.strip() != "ID=0"]

            if len(new_lines) != len(lines):
                modified_files.append(tps_file_name)
                if not dry_run:
                    with open(tps_file_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)

        except Exception as e:
            print(f"Error processing '{tps_file_name}': {e}")

    # Summary Output
    print("\n--- Analysis & Modification Summary ---")
    print(f"Total TPS files scanned: {len(tps_files)}")

    if files_without_scale:
        print(
            f"Files MISSING 'SCALE=' ({len(files_without_scale)}): {files_without_scale}"
        )
    else:
        print("All files contain 'SCALE='.")

    if files_without_lm12:
        print(
            f"Files MISSING 'LM=12' ({len(files_without_lm12)}): {files_without_lm12}"
        )
    else:
        print("All files contain 'LM=12'.")

    if modified_files:
        action = "WOULD BE MODIFIED" if dry_run else "MODIFIED"
        print(f"Files where 'ID=0' was removed ({action}): {modified_files}")
    else:
        print("No files required 'ID=0' removal.")

    return {
        "total": len(tps_files),
        "missing_scale": files_without_scale,
        "missing_lm12": files_without_lm12,
        "modified": modified_files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify landmark TPS files and clean ID=0 entries."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/tps",
        help="Path to directory containing TPS files (default: ./data/tps)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="h",
        choices=["h", "f"],
        help="TPS file suffix character ('h' for hindwing, 'f' for forewing)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without overwriting files",
    )

    args = parser.parse_args()
    analyze_and_modify_tps(args.dir, args.suffix, args.dry_run)
