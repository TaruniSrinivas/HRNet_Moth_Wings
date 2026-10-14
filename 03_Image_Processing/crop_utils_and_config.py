# ==============================================================================
# SCRIPT PURPOSE: Configuration paths, file extension defaults, and helper utilities 
#                 for computing enclosing bounding boxes and normalizing Y-coordinates.
# INPUTS:         Invoked as a module dependency by cropping and dataset preparation scripts.
# OUTPUTS:        Helper math functions: calculate_min_enclosing_bbox(), normalize_y_coordinates()
# PAPER SECTION:  Methods - Section 2.2 (Image Coordinate Normalization & Utilities)
# ==============================================================================

import os

# --- Default Relative Repository Paths ---
DEFAULT_IMAGE_INPUT_DIR = "./data/images"
DEFAULT_OUTPUT_BASE_DIR = "./data/cropped_output"
DEFAULT_CSV_DIR = "./data"

# Subdirectories for outputs
DEFAULT_RAW_CROPPED_DIR = os.path.join(
    DEFAULT_OUTPUT_BASE_DIR, "01_Cropped_wings"
)
DEFAULT_PREPROCESSED_DIR = os.path.join(
    DEFAULT_OUTPUT_BASE_DIR, "02_preprocessed_images"
)
DEFAULT_BBOX_OUTLINED_DIR = os.path.join(
    DEFAULT_OUTPUT_BASE_DIR, "03_bbox_outlined_images"
)

# Coordinates and Output CSVs
DEFAULT_BBOX_COORDS_CSV = os.path.join(DEFAULT_CSV_DIR, "bbox_coords.csv")
DEFAULT_12_POINTS_CSV = os.path.join(
    DEFAULT_CSV_DIR, "Hindwing_Test_Geometrids.csv"
)

IMAGE_FILE_EXTENSIONS = [
    "*.jpg",
    "*.JPG",
    "*.png",
    "*.PNG",
    "*.JPEG",
    "*.jpeg",
]
DEFAULT_PADDING_PERCENTAGE = 0.02


def calculate_min_enclosing_bbox(points, image_shape, padding_percentage=0.02):
    """Calculates the minimum axis-aligned bounding box enclosing a list of points

    with padding applied as a percentage of initial box dimensions.
    """
    if not points:
        print("Error: No points provided for bounding box calculation.")
        return None

    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]

    min_col_initial = int(min(all_x))
    max_col_initial = int(max(all_x))
    min_row_initial = int(min(all_y))
    max_row_initial = int(max(all_y))

    initial_width = max_col_initial - min_col_initial
    initial_height = max_row_initial - min_row_initial

    padding_x_px = int(initial_width * padding_percentage)
    padding_y_px = int(initial_height * padding_percentage)

    min_row_padded = min_row_initial - padding_y_px
    max_row_padded = max_row_initial + padding_y_px
    min_col_padded = min_col_initial - padding_x_px
    max_col_padded = max_col_initial + padding_x_px

    image_height, image_width = image_shape[:2]

    # Clamp coordinates to image bounds
    min_row_clamped = max(0, min_row_padded)
    max_row_clamped = min(image_height - 1, max_row_padded)
    min_col_clamped = max(0, min_col_padded)
    max_col_clamped = min(image_width - 1, max_col_padded)

    if max_col_clamped <= min_col_clamped:
        max_col_clamped = min_col_clamped + 1
    if max_row_clamped <= min_row_clamped:
        max_row_clamped = min_row_clamped + 1

    return (min_row_clamped, min_col_clamped, max_row_clamped, max_col_clamped)


def normalize_y_coordinates(points, image_height):
    """Converts Y-coordinates from Cartesian/TPS layout to standard image coordinate indexing."""
    normalized_points = []
    for x, y in points:
        normalized_y = (image_height - 1) - y
        normalized_points.append((x, normalized_y))
    return normalized_points
