# ==============================================================================
# SCRIPT PURPOSE: PyTorch Dataset class (WingLandmarksDataset) that loads COCO JSON
#                 annotations, flips Y-coordinates to image space, crops specimens,
#                 and constructs target Gaussian heatmaps for keypoint training.
# INPUTS:         json_file : Path to COCO annotation JSON file
#                 cfg       : YACS configuration node
# OUTPUTS:        PyTorch DataLoader tuples: (normalized_image, target_heatmaps, meta_dict)
# PAPER SECTION:  Methods - Section 2.3 & 2.4 (Dataset Parsing & Target Heatmap Generation)
# ==============================================================================

import os 
import torch 
import torchvision.transforms as transforms 
from torchvision.transforms import functional as F 
from pycocotools.coco import COCO 
from PIL import Image 
import numpy as np 
from utils import generate_target, crop, transform_pixel 

class WingLandmarksDataset(torch.utils.data.Dataset): 
    def __init__(self, json_file, cfg): 
        self.root_dir = cfg.DATASET.ROOT 
        self.transforms = None 
        self.coco = COCO(json_file) 
        self.image_ids = self.coco.getImgIds() 
        self.num_landmarks = cfg.DATASET.NUM_LANDMARKS 
        self.padding = cfg.DATASET.PADDING 
        self.sigma = cfg.MODEL.SIGMA 
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE 
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) 
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32) 
        print("Dataset initialized with Coordinate Flip Fix.") 

    def __len__(self): 
        return len(self.image_ids) 

    def __getitem__(self, idx): 
        img_id = self.image_ids[idx] 
        image_info = self.coco.loadImgs(img_id)[0] 
        img_path = os.path.join(self.root_dir, image_info['file_name']) 

        pil_image = Image.open(img_path).convert('RGB') 
        image = np.array(pil_image) 
        w, h = pil_image.size
        
        # Load annotations 
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Parse Keypoints
        keypoints = np.zeros((self.num_landmarks, 3), dtype=np.float32) 
        if annotations and 'keypoints' in annotations[0]: 
            raw_keypoints_list = annotations[0]['keypoints']
            # FIX: Ensure length matches before reshaping to prevent ValueError
            if len(raw_keypoints_list) == self.num_landmarks * 3:
                keypoints = np.array(raw_keypoints_list, dtype=np.float32).reshape(self.num_landmarks, 3)
            else:
                print(f"Warning: Annotation for image {img_id} has {len(raw_keypoints_list)} keypoint values. Expected {self.num_landmarks * 3}. Using zeros fallback.")

            # --- CRITICAL FIX: FLIP Y COORDINATES ---
            # Convert from Bottom-Left Origin (TPSDig) to Top-Left Origin (OpenCV)
            # Only flip visible keypoints (v > 0) or all if you treat v=0 as valid coords
            # Assuming all keypoints in JSON are valid coordinates:
            keypoints[:, 1] = h - keypoints[:, 1]
            # ----------------------------------------
        
        # --- DYNAMIC CENTERING (Using Flipped Coordinates) ---
        # Calculate center/scale from the FLIPPED keypoints
        # This ensures the crop centers on the wing's actual pixel location
        if np.sum(keypoints[:, 2] > 0) > 0: # Check if there are valid points
             valid_kps = keypoints # Use all for wing calculation
             
             min_x = np.min(valid_kps[:, 0])
             max_x = np.max(valid_kps[:, 0])
             min_y = np.min(valid_kps[:, 1])
             max_y = np.max(valid_kps[:, 1])
             
             center_x = (min_x + max_x) / 2.0
             center_y = (min_y + max_y) / 2.0
             center = np.array([center_x, center_y], dtype=np.float32)
             
             width = max_x - min_x
             height = max_y - min_y
             scale = max(width, height) / 200.0 * self.padding
        else:
             # Fallback
             center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
             scale = max(w, h) / 200.0 * self.padding
        
        origin_size = torch.tensor([h, w]) 
        
        r = 0 
        image = crop(image, center, scale, [256, 256], rot=r) 
        
        target = np.zeros((self.num_landmarks, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32) 
        tpts = np.zeros((self.num_landmarks, 2)) 

        for i in range(self.num_landmarks): 
            tpts[i, 0:2] = transform_pixel(keypoints[i, 0:2], center, scale, self.heatmap_size, rot=r) 
            target[i] = generate_target(target[i], tpts[i], self.sigma) 

        image = image.astype(np.float32) 
        image = (image - self.mean) / self.std 
        image = image.transpose([2, 0, 1]) 
        target = torch.Tensor(target) 
        tpts = torch.Tensor(tpts) 

        meta = {'index': idx, 'pts': keypoints, 'center': center, 'scale': scale, 'size': origin_size} 

        return image, target, meta
