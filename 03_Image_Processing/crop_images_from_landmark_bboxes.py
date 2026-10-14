# ==============================================================================
# SCRIPT PURPOSE: Batch crops specimen images using a minimum bounding box calculated
#                 from 12 landmark coordinates, with optional percentage padding.
# INPUTS:         --image_dir       : Path to input images directory (default: ./data/images)
#                 --coords_csv      : CSV file with landmark coordinates (X1-X12, Y1-Y12)
# OUTPUTS:        --output_crop_dir : Directory to save cropped specimen images
#                 --output_bbox_csv : Output CSV saving bounding box spatial dimensions
# PAPER SECTION:  Methods - Section 2.2 (Specimen Bounding Box Generation & Image Cropping)
# ==============================================================================

import argparse
from glob import glob
import os
from config_and_utils import (
    DEFAULT_12_POINTS_CSV,
    DEFAULT_BBOX_COORDS_CSV,
    DEFAULT_IMAGE_INPUT_DIR,
    DEFAULT_PADDING_PERCENTAGE,
    DEFAULT_RAW_CROPPED_DIR,
    IMAGE_FILE_EXTENSIONS,
    calculate_min_enclosing_bbox,
    normalize_y_coordinates,
)
import numpy as np
import pandas as pd

try:
    import skimage.io as io
except ImportError:
    from PIL import Image

    class ImageIOShim:

        @staticmethod
        def imread(path):
            return np.array(Image.open(path))

        @staticmethod
        def imsave(path, arr):
            Image.fromarray(arr).save(path)

    io = ImageIOShim()


def batch_crop_multiple_points(
    image_dir=DEFAULT_IMAGE_INPUT_DIR,
    coords_csv_path=DEFAULT_12_POINTS_CSV,
    output_crop_dir=DEFAULT_RAW_CROPPED_DIR,
    output_bbox_csv_path=DEFAULT_BBOX_COORDS_CSV,
    padding_pct=DEFAULT_PADDING_PERCENTAGE,
):
    print("\n--- Starting Batch Cropping (Preserving CSV Order) ---")

    os.makedirs(output_crop_dir, exist_ok=True)
    calculated_bbox_data = []

    try:
        coords_df = pd.read_csv(coords_csv_path)
        print(
            f"Loaded landmark coordinates for {len(coords_df)} images from '{coords_csv_path}'."
        )
    except Exception as e:
        print(f"Error reading coordinates CSV file '{coords_csv_path}': {e}")
        return False

    available_images = {}
    for ext_pattern in IMAGE_FILE_EXTENSIONS:
        for filepath in glob(os.path.join(image_dir, ext_pattern)):
            available_images[os.path.basename(filepath)] = filepath

    if not available_images:
        print(f"No image files found in '{image_dir}'.")
        return False

    processed_count = 0
    skipped_count = 0

    for index, row in coords_df.iterrows():
        img_name = row.get("File_name")

        if pd.isna(img_name) or img_name not in available_images:
            skipped_count += 1
            continue

        img_full_path = available_images[img_name]
        name_without_ext, ext = os.path.splitext(img_name)

        try:
            image = io.imread(img_full_path)
            if image is None:
                skipped_count += 1
                continue

            image_height = image.shape[0]

            original_12_points = []
            for i in range(1, 13):
                x = row[f"X{i}"]
                y = row[f"Y{i}"]
                original_12_points.append((x, y))

            normalized_12_points = normalize_y_coordinates(
                original_12_points, image_height
            )
            bbox_dims = calculate_min_enclosing_bbox(
                normalized_12_points,
                image.shape,
                padding_percentage=padding_pct,
            )

            if bbox_dims is None:
                skipped_count += 1
                continue

            min_row, min_col, max_row, max_col = bbox_dims
            bbox_height = max_row - min_row + 1
            bbox_width = max_col - min_col + 1

            calculated_bbox_data.append(
                {
                    "image_name": img_name,
                    "min_col": min_col,
                    "min_row": min_row,
                    "max_col": max_col,
                    "max_row": max_row,
                    "bbox_height": bbox_height,
                    "bbox_width": bbox_width,
                    "x_top_left": min_col,
                    "y_top_left": min_row,
                }
            )

            cropped_image = image[
                min_row : max_row + 1, min_col : max_col + 1
            ]
            output_filename = f"{name_without_ext}_cropped{ext}"
            output_path = os.path.join(output_crop_dir, output_filename)
            io.imsave(output_path, cropped_image)

            processed_count += 1

        except Exception as e:
            print(f"Error processing '{img_name}': {e}")
            skipped_count += 1
            continue

    if calculated_bbox_data:
        bbox_df = pd.DataFrame(calculated_bbox_data)
        os.makedirs(
            os.path.dirname(os.path.abspath(output_bbox_csv_path)),
            exist_ok=True,
        )
        bbox_df.to_csv(output_bbox_csv_path, index=False)
        print(
            f"Successfully saved calculated bounding box coordinates to: '{output_bbox_csv_path}'"
        )

    print(f"\n--- Batch Cropping Complete ---")
    print(f"Processed: {processed_count} | Skipped: {skipped_count}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch crop specimens based on 12 landmark points."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_INPUT_DIR,
        help="Path to directory containing input images",
    )
    parser.add_argument(
        "--coords_csv",
        type=str,
        default=DEFAULT_12_POINTS_CSV,
        help="Path to CSV file with landmark coordinates",
    )
    parser.add_argument(
        "--output_crop_dir",
        type=str,
        default=DEFAULT_RAW_CROPPED_DIR,
        help="Directory to save cropped images",
    )
    parser.add_argument(
        "--output_bbox_csv",
        type=str,
        default=DEFAULT_BBOX_COORDS_CSV,
        help="Path to save calculated bounding box coordinates",
    )
    parser.add_argument(
        "--padding_pct",
        type=float,
        default=DEFAULT_PADDING_PERCENTAGE,
        help="Padding percentage around bounding box",
    )

    args = parser.parse_args()
    batch_crop_multiple_points(
        image_dir=args.image_dir,
        coords_csv_path=args.coords_csv,
        output_crop_dir=args.output_crop_dir,
        output_bbox_csv_path=args.output_bbox_csv,
        padding_pct=args.padding_pct,
    )
