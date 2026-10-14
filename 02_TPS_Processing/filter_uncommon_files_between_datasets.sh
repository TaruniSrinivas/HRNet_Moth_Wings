#!/bin/bash
# ==============================================================================
# SCRIPT PURPOSE: Cross-references specimen image files in a target folder against
#                 a keep-list text file and deletes unlisted files to keep datasets aligned.
# INPUTS:         $1 : Target image directory (default: ./data/images)
#                 $2 : Path to keep-list text file (default: ./data/extra_filenames.txt)
#                 $3 : Safety flag DRY_RUN (true=preview, false=delete unlisted files)
# OUTPUTS:        Console summary of retained and deleted files; pruned directory state
# PAPER SECTION:  Methods - Section 2.1 (Dataset Cross-Referencing & Pruning)
# ==============================================================================

!/bin/bash

set -euo pipefail

# Configuration with reviewer defaults
TARGET_DIR="${1:-./data/images}"
KEEP_LIST="${2:-./data/extra_filenames.txt}"
DRY_RUN="${3:-true}"  # Default to safety mode (true) to prevent accidental data loss

# Check if keep list exists
if [[ ! -f "$KEEP_LIST" ]]; then
    echo "Error: Keep list file not found at '$KEEP_LIST'" >&2
    exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Target directory not found at '$TARGET_DIR'" >&2
    exit 1
fi

retained_count=0
deleted_count=0

echo "Starting filtering process in: $TARGET_DIR"
echo "Cross-referencing against: $KEEP_LIST"

if [ "$DRY_RUN" = true ]; then
    echo "--- SAFETY MODE ACTIVE (Dry Run: No files will be deleted) ---"
else
    echo "--- WARNING: EXECUTION MODE (Files NOT on list will be DELETED) ---"
fi
echo "------------------------------------------"

# Expand pattern safely without failing if empty
shopt -s nullglob
files=("$TARGET_DIR"/*_AA[0-9][0-9][0-9][0-9]a_Orig*)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "No matching files found in $TARGET_DIR"
    exit 0
fi

for file_path in "${files[@]}"; do
    if [[ -f "$file_path" ]]; then
        file_name=$(basename "$file_path")
        
        # Extract ID (e.g., AA0892)
        if [[ $file_name =~ (AA[0-9]{4}) ]]; then
            code="${BASH_REMATCH[1]}"
            
            # Check if ID exists in list (stripping Windows \r line endings)
            if grep -qw "$code" <(sed 's/\r$//' "$KEEP_LIST"); then
                ((retained_count++))
                echo "KEEPING  [$retained_count]: $file_name"
            else
                ((deleted_count++))
                if [ "$DRY_RUN" = true ]; then
                    echo "WOULD DELETE: $file_name"
                else
                    echo "DELETING: $file_name"
                    rm "$file_path"
                fi
            fi
        fi
    fi
done

echo "------------------------------------------"
echo "PROCESS SUMMARY:"
echo "Files Retained: $retained_count"
echo "Files Deleted:  $deleted_count"
echo "------------------------------------------"

if [ "$DRY_RUN" = true ]; then
    echo "To perform actual deletion, pass 'false' as the 3rd argument:"
    echo "  bash filter_files.sh <target_dir> <keep_list> false"
fi
