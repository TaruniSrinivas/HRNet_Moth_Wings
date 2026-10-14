# ==============================================================================
# SCRIPT PURPOSE: Math transform functions (affine cropping, pixel space inversion),
#                 heatmap decoding, logger instantiation, and NME evaluation calculations.
# INPUTS:         Invoked as helper library by training, dataset, and inference modules
# OUTPUTS:        Decoded keypoint coordinates, affine warped images, and evaluation metrics
# PAPER SECTION:  Methods - Section 2.4 & 2.6 (Spatial Transformations & Metric Calculations)
# ==============================================================================

import os
import math
import numpy as np
import torch
import cv2  # Requires pip install opencv-python
import logging
from pathlib import Path
import time

# ------------------------------------------------------------------------------
# MATH & TRANSFORMATION FUNCTIONS (THE CORE FIXES)
# ------------------------------------------------------------------------------

def get_transform(center, scale, output_size, rot=0):
    """
    Standard affine transformation matrix for HRNet.
    """
    h = 200 * scale
    w = h
    target_w, target_h = output_size

    # Matrix to center and scale
    t = np.zeros((3, 3))
    t[0, 0] = float(target_w) / w
    t[1, 1] = float(target_h) / h
    t[0, 2] = target_w * (-float(center[0]) / w + 0.5)
    t[1, 2] = target_h * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1

    if rot != 0:
        rot_rad = -rot * np.pi / 180.0
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat = np.eye(3)
        rot_mat[0, 0] = cs
        rot_mat[0, 1] = -sn
        rot_mat[1, 0] = sn
        rot_mat[1, 1] = cs
        
        # Rotate around the center of the output image
        t_center = np.eye(3)
        t_center[0, 2] = -target_w / 2
        t_center[1, 2] = -target_h / 2
        
        t_inv_center = np.eye(3)
        t_inv_center[0, 2] = target_w / 2
        t_inv_center[1, 2] = target_h / 2
        
        t = np.dot(t_inv_center, np.dot(rot_mat, np.dot(t_center, t)))

    return t

def crop(img, center, scale, output_size, rot=0):
    """
    Robust crop using cv2.warpAffine.
    """
    trans = get_transform(center, scale, output_size, rot)
    trans_cv = trans[:2, :] # cv2 expects 2x3
    
    dst_img = cv2.warpAffine(
        img, 
        trans_cv, 
        (int(output_size[0]), int(output_size[1])), 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return dst_img

def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    """
    Transforms a single point. Handles both Tensor and Numpy inputs safely.
    """
    # Safety: Convert inputs to numpy/scalar if they are Tensors
    pt_np = pt.cpu().numpy() if torch.is_tensor(pt) else pt
    center_np = center.cpu().numpy() if torch.is_tensor(center) else center
    scale_np = scale.item() if torch.is_tensor(scale) else scale

    t = get_transform(center_np, scale_np, output_size, rot)
    if invert:
        t = np.linalg.inv(t)

    new_pt = np.array([pt_np[0], pt_np[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = transform_pixel(coords[p, 0:2], center, scale, output_size, invert=1, rot=0)
    return target_coords


# ------------------------------------------------------------------------------
# LOGGING & HELPERS
# ------------------------------------------------------------------------------

def create_logger(cfg, cfg_name, phase='train'):
    if phase=='train': root_output_dir = Path('./train_dirs') 
    elif phase=='predict': root_output_dir = Path('./predict_dirs') 
    
    if not root_output_dir.exists(): 
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir() 

    dataset = cfg.DATASET.DATASET 
    model = cfg.MODEL.NAME 
    cfg_name = os.path.basename(cfg_name) 

    final_output_dir = root_output_dir / dataset / cfg_name 
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True) 

    time_str = time.strftime('%Y-%m-%d-%H-%M') 
    log_file = '{}.log'.format(time_str) 
    final_log_file = final_output_dir / log_file 
    head = '%(asctime)-15s %(message)s' 
    logging.basicConfig(filename=str(final_log_file), format=head) 
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) 
    console = logging.StreamHandler() 
    logging.getLogger('').addHandler(console) 

    tensorboard_log_dir = final_output_dir / ('event.tb.' + time_str) 
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True) 

    return logger, str(final_output_dir), str(tensorboard_log_dir)

class AverageMeter(object): 
    def __init__(self): self.reset() 
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0 
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


# ------------------------------------------------------------------------------
# HEATMAP & EVALUATION
# ------------------------------------------------------------------------------

def generate_target(img, pt, sigma, label_type='Gaussian'): 
    tmp_size = sigma * 3 
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)] 
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)] 
    
    heatmap_h, heatmap_w = img.shape[0], img.shape[1] 

    if (ul[0] >= heatmap_w or ul[1] >= heatmap_h or br[0] < 0 or br[1] < 0): 
        return img 

    size = 2 * tmp_size + 1 
    x = np.arange(0, size, 1, np.float32) 
    y = x[:, np.newaxis] 
    x0 = y0 = size // 2 

    if label_type == 'Gaussian': 
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) 
    else: 
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5) 

    g_x = max(0, -ul[0]), min(br[0], heatmap_w) - ul[0] 
    g_y = max(0, -ul[1]), min(br[1], heatmap_h) - ul[1] 
    img_x = max(0, ul[0]), min(heatmap_w, br[0]) 
    img_y = max(0, ul[1]), min(heatmap_h, br[1]) 

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]] 
    return img 

def get_preds(scores): 
    assert scores.dim() == 4, 'Score maps should be 4-dim' 
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2) 

    maxval = maxval.view(scores.size(0), scores.size(1), 1) 
    idx = idx.view(scores.size(0), scores.size(1), 1) 

    preds = idx.repeat(1, 1, 2).float() 

    preds[:, :, 0] = preds[:, :, 0] % scores.size(3) 
    preds[:, :, 1] = preds[:, :, 1] // scores.size(3) 

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float() 
    preds *= pred_mask 
    return preds 

def decode_preds(output, center, scale, heatmap_size): 
    coords = get_preds(output) 
    
    # Refine predictions with quarter-pixel shift
    for n in range(coords.size(0)): 
        for p in range(coords.size(1)): 
            hm = output[n][p] 
            px = int(math.floor(coords[n][p][0])) 
            py = int(math.floor(coords[n][p][1])) 
            if (px > 0 and px < heatmap_size[0] - 1) and (py > 0 and py < heatmap_size[1] - 1): 
                dx = 0.25 if hm[py, px + 1] > hm[py, px - 1] else -0.25 
                dy = 0.25 if hm[py + 1, px] > hm[py - 1, px] else -0.25 
                coords[n][p][0] += dx 
                coords[n][p][1] += dy 
    
    coords += 0.5 
    
    # --- FIX: Convert to NumPy array HERE ---
    # This prevents the TypeError when assigning the result of transform_preds
    preds = coords.cpu().numpy() 

    for i in range(coords.size(0)): 
        preds[i] = transform_preds(coords[i], center[i], scale[i], heatmap_size) 
        
    return preds 

def compute_nme(preds, pts_gt): 
    if preds.ndim == 2 and pts_gt.ndim == 2:
        preds = np.expand_dims(preds, axis=0)
        pts_gt = np.expand_dims(pts_gt, axis=0)

    try:
        wing_length = np.linalg.norm(pts_gt[:, 11, :] - pts_gt[:, 4, :], axis=1) 
    except IndexError:
        wing_length = np.linalg.norm(pts_gt[:, 0, :] - pts_gt[:, 1, :], axis=1)
        
    wing_length[wing_length == 0] = 1e-6 
    nme_per_point = np.linalg.norm(preds - pts_gt, axis=2) / wing_length[:, np.newaxis] 
    return nme_per_point.mean(axis=1), np.linalg.norm(preds - pts_gt, axis=2).mean(axis=1)

def compute_nme_on_landmark(preds, pts): 
    if preds.ndim == 2 and pts.ndim == 2:
        preds = np.expand_dims(preds, axis=0)
        pts = np.expand_dims(pts, axis=0)

    try:
        wing_length = np.linalg.norm(pts[:, 11, :] - pts[:, 4, :], axis=1)
    except IndexError:
        wing_length = np.linalg.norm(pts[:, 0, :] - pts[:, 1, :], axis=1)

    wing_length[wing_length == 0] = 1e-6

    nme_lms = np.linalg.norm(preds - pts, axis=2) / wing_length[:, np.newaxis]
    nme_lms_abs = np.linalg.norm(preds - pts, axis=2)

    return nme_lms, nme_lms_abs


# ------------------------------------------------------------------------------
# CHECKPOINTING
# ------------------------------------------------------------------------------

def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    """
    Saves model checkpoint.
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))