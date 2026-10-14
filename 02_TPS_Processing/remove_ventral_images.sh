#!/bin/bash
# ==============================================================================
# SCRIPT PURPOSE: Identifies and removes ventral view specimen images (filenames ending 
#                 with 'b') while preserving dorsal view images (ending with 'a').
# INPUTS:         $1 : Target image directory (default: ./data/images)
#                 $2 : Safety flag DRY_RUN (default: true for dry run, false for deletion)
# OUTPUTS:        Console audit summary; cleans target directory of unwanted ventral files
# PAPER SECTION:  Methods - Section 2.1 (Dorsal vs. Ventral Image Filtering)
# ==============================================================================

!/bin/bash

set -euo pipefail

# Configuration with relative reviewer defaults
TARGET_DIR="${1:-./data/images}"
DRY_RUN="${2:-true}"  # Default to safety mode (true) to prevent accidental data loss

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory '$TARGET_DIR' not found." >&2
    exit 1
fi

echo "Starting cleanup of 'b' files in: $TARGET_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "--- SAFETY MODE ACTIVE (Dry Run: No files will be deleted) ---"
else
    echo "--- WARNING: EXECUTION MODE (Matching 'b' files will be DELETED) ---"
fi
echo "----------------------------------------------------"

# Expand pattern safely without failing if empty
shopt -s nullglob
files=("$TARGET_DIR"/*_AA[0-9][0-9][0-9][0-9]b_Orig*)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "No matching 'b' files found in $TARGET_DIR"
    exit 0
fi

deleted_count=0

for file_path in "${files[@]}"; do
    if [[ -f "$file_path" ]]; then
        file_name=$(basename "$file_path")
        ((deleted_count++))
        
        if [ "$DRY_RUN" = true ]; then
            echo "WOULD DELETE [$deleted_count]: $file_name"
        else
            echo "DELETING [$deleted_count]: $file_name"
            rm "$file_path"
        fi
    fi
done

echo "----------------------------------------------------"
echo "PROCESS SUMMARY:"
echo "Total 'b' files processed: $deleted_count"
echo "All 'a' files were preserved."
echo "----------------------------------------------------"

if [ "$DRY_RUN" = true ]; then
    echo "To perform actual deletion, pass 'false' as the 2nd argument:"
    echo "  bash remove_b_files.sh <target_dir> false"
fi
