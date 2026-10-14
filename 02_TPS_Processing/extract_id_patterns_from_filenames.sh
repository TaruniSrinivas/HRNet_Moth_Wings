#!/bin/bash
# ==============================================================================
# SCRIPT PURPOSE: Searches specimen image filenames in a directory for specific ID regex
#                 patterns (e.g., AA0000) and exports extracted codes to a text file.
# INPUTS:         $1 : Target directory containing images (default: ./data/images)
#                 $2 : Output text file path (default: ./data/extracted_codes.txt)
#                 $3 : Regex pattern to extract (default: AA[0-9]{4})
# OUTPUTS:        Text file containing line-separated extracted specimen codes
# PAPER SECTION:  Methods - Section 2.1 (Specimen Identification & Code Extraction)
# ==============================================================================

!/bin/bash

set -euo pipefail

# Default parameters target the repository's data folder and paper's pattern
TARGET_DIRECTORY="${1:-./data/images}"
OUTPUT_FILE="${2:-./data/extracted_codes.txt}"
PATTERN="${3:-AA[0-9]{4}}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
> "$OUTPUT_FILE"

if [ ! -d "$TARGET_DIRECTORY" ]; then
    echo "Error: Directory '$TARGET_DIRECTORY' not found." >&2
    exit 1
fi

echo "Processing files in: $TARGET_DIRECTORY"
echo "Extracting pattern '$PATTERN' -> $OUTPUT_FILE"
echo "-----------------------------------"

find "$TARGET_DIRECTORY" -maxdepth 1 -type f -print0 | while IFS= read -r -d $'\0' filename; do
    base_filename=$(basename "$filename")
    extracted_code=$(echo "$base_filename" | grep -oP "$PATTERN" || true)

    if [ -n "$extracted_code" ]; then
        echo "$extracted_code" >> "$OUTPUT_FILE"
        echo "Processed: $base_filename -> Extracted: $extracted_code"
    else
        echo "Processed: $base_filename -> No matching pattern found."
    fi
done

echo "-----------------------------------"
echo "Done. Extracted codes saved to $OUTPUT_FILE"
