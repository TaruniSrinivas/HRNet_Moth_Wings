#!/bin/bash
# ==============================================================================
# SCRIPT PURPOSE: Recursively searches subdirectories within a source folder and copies
#                 all image files into a single flattened destination folder.
# INPUTS:         $1 : Source directory with nested folders (default: ./data/nested_images)
#                 $2 : Destination directory for flat storage (default: ./data/all_images_flat)
# OUTPUTS:        Flat destination directory containing all copied files and total file count
# PAPER SECTION:  Methods - Section 2.1 (Directory Structure Standardization)
# ==============================================================================

!/bin/bash

set -euo pipefail

# Configuration with relative reviewer defaults
SOURCE_DIR="${1:-./data/nested_images}"
DEST_DIR="${2:-./data/all_images_flat}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found." >&2
    exit 1
fi

# Create destination directory if it does not exist
mkdir -p "$DEST_DIR"

echo "Extracting and flattening files from: $SOURCE_DIR"
echo "Copying to destination: $DEST_DIR"
echo "----------------------------------------------------"

# Copy all nested files into flat target directory
find "$SOURCE_DIR" -type f -exec cp -t "$DEST_DIR" {} +

TOTAL_COPIED=$(find "$DEST_DIR" -maxdepth 1 -type f | wc -l)

echo "----------------------------------------------------"
echo "Done! Check $DEST_DIR"
echo "Total files in flattened directory: $TOTAL_COPIED"
