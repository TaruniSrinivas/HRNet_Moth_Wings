# ==============================================================================
# SCRIPT PURPOSE: Generates visual quality control overlays comparing Ground Truth 
#                 (Human, Red) vs HRNet Model Predicted (Machine, Blue) keypoints on images.
# INPUTS:         --json     : COCO test annotations JSON file
#                 --img_dir  : Directory containing target specimen images
#                 --preds    : PyTorch predictions file (.pth) containing model output keypoints
# OUTPUTS:        --out_dir  : Directory to save visual comparison check images
# PAPER SECTION:  Methods - Section 2.6 (Model Evaluation & Qualitative Landmark Validation)
# ==============================================================================

import argparse
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import torch


def visualize_landmark_predictions(
    json_path,
    img_dir,
    preds_path,
    output_dir,
    num_samples=20,
    flip_y_gt=True,
    flip_y_pred=False,
):
    """Overlays Ground Truth (Red) and Predicted (Blue) landmarks onto specimen images."""
    if not os.path.exists(json_path):
        print(f"Error: COCO JSON file not found at '{json_path}'")
        return False
    if not os.path.exists(preds_path):
        print(f"Error: Predictions file not found at '{preds_path}'")
        return False
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at '{img_dir}'")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Ground Truth Annotations
    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    # 2. Load Prediction Weights/Tensors
    preds_data = torch.load(preds_path, map_location="cpu", weights_only='True')
    if isinstance(preds_data, dict) and "preds" in preds_data:
        predictions = preds_data["preds"]
    else:
        predictions = preds_data

    if torch.is_tensor(predictions):
        predictions = predictions.numpy()

    total_to_process = min(num_samples, len(img_ids))
    print(f"Generating visual checks for {total_to_process} images...")

    for i in range(total_to_process):
        img_id = img_ids[i]
        img_info = coco.loadImgs(img_id)[0]
        fname = img_info["file_name"]
        img_path = os.path.join(img_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image at '{img_path}'. Skipping.")
            continue

        h, w = img.shape[:2]

        # --- DRAW GROUND TRUTH (Red Circles) ---
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if anns and "keypoints" in anns[0]:
            gt_kps = np.array(anns[0]["keypoints"], dtype=np.float32).reshape(
                -1, 3
            )
            for kp in gt_kps:
                x, y, v = kp
                if v > 0:
                    y_plot = (h - y) if flip_y_gt else y
                    cv2.circle(
                        img,
                        (int(x), int(y_plot)),
                        6,
                        (0, 0, 255),
                        -1,
                    )

        # --- DRAW PREDICTIONS (Blue Circles) ---
        if i < len(predictions):
            pred_kps = predictions[i]
            for kp in pred_kps:
                x, y = kp[:2]
                y_plot = (h - y) if flip_y_pred else y
                cv2.circle(
                    img,
                    (int(x), int(y_plot)),
                    5,
                    (255, 0, 0),
                    -1,
                )

        # --- ON-IMAGE LEGEND ---
        cv2.putText(
            img,
            "Red: Ground Truth",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            img,
            "Blue: HRNet Prediction",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        out_name = os.path.join(output_dir, f"vis_check_{i:03d}_{fname}")
        cv2.imwrite(out_name, img)

    print(f"Successfully saved visualization images to: '{output_dir}'")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Ground Truth vs HRNet Predicted Keypoints."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="./data/test_annotations.json",
        help="Path to COCO test annotations JSON",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="./data/images",
        help="Path to directory containing test images",
    )
    parser.add_argument(
        "--preds",
        type=str,
        default="./predict_dirs/predictions.pth",
        help="Path to predictions.pth file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./predict_dirs/visual_checks",
        help="Output directory for annotated preview images",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of images to generate preview overlays for",
    )
    parser.add_argument(
        "--flip_y_gt",
        action="store_true",
        default=True,
        help="Flip GT Y-coordinates from TPSDig bottom-left origin",
    )
    parser.add_argument(
        "--flip_y_pred",
        action="store_true",
        default=False,
        help="Flip predicted Y-coordinates if model was trained on inverted space",
    )

    args = parser.parse_args()
    visualize_landmark_predictions(
        json_path=args.json,
        img_dir=args.img_dir,
        preds_path=args.preds,
        output_dir=args.out_dir,
        num_samples=args.num_samples,
        flip_y_gt=args.flip_y_gt,
        flip_y_pred=args.flip_y_pred,
    )
