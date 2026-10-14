# ==============================================================================
# SCRIPT PURPOSE: Evaluates a trained HRNet checkpoint on test datasets, computing
#                 Normalized Mean Error (NME) metrics and exporting prediction files.
# INPUTS:         --cfg : Path to evaluation YAML configuration file (e.g., hrnet_predict.yaml)
# OUTPUTS:        predictions.pth (tensor predictions) and nme_on_landmarks.csv (per-landmark NME)
# PAPER SECTION:  Methods - Section 2.6 (Model Testing, NME Evaluation & Inference)
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import pprint
import time
import os

import models
from config import update_config
from config import _C as config
from dataset import WingLandmarksDataset
from utils import get_preds, decode_preds, compute_nme, compute_nme_on_landmark
from utils import AverageMeter, create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Predict wing landmarks')
    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    return args


def inference(config, data_loader, model, device, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    num_landmarks = config.MODEL.NUM_JOINTS 
    predictions = torch.zeros((len(data_loader.dataset), num_landmarks, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    nme_abs_batch_sum = 0
    
    # Initialize landmark-wise NME sums (for 12 landmarks)
    nme_lms_batch_sum_total = np.zeros(num_landmarks) 
    nme_lms_abs_batch_sum_total = np.zeros(num_landmarks)
    
    count_failure_001 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inp = inp.to(device)
            
            # FIXED: Convert pts_gt to NumPy for compute_nme (as expected by utils.py)
            pts_gt = meta['pts'][:, :, 0:2].cpu().numpy() 
            
            # ALIGNMENT: Use autocast 
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                 output = model(inp)

            score_map = output.data.cpu() 
            
            # FIXED: Use config.MODEL.HEATMAP_SIZE
            preds = decode_preds(
                score_map,
                meta['center'],
                meta['scale'],
                config.MODEL.HEATMAP_SIZE 
            )

            # NME
            nme_temp, nme_abs_temp = compute_nme(preds, pts_gt)
            # NME on landmarks
            nme_lms_temp, nme_lms_abs_temp = compute_nme_on_landmark(preds, pts_gt)

            failure_001 = (nme_temp > 0.01).sum()
            count_failure_001 += failure_001

            nme_batch_sum += np.sum(nme_temp)
            nme_abs_batch_sum += np.sum(nme_abs_temp)
            
            # Accumulate landmark-wise NME correctly
            nme_lms_batch_sum_total += np.sum(nme_lms_temp, axis=0)
            nme_lms_abs_batch_sum_total += np.sum(nme_lms_abs_temp, axis=0)
            
            nme_count = nme_count + preds.shape[0] 
            for n in range(score_map.size(0)):
                # FIXED: Convert NumPy preds back to tensor for assignment
                predictions[meta['index'][n], :, :] = torch.from_numpy(preds[n, :, :]) 

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # Final calculations, robust against zero count
    nme = nme_batch_sum / nme_count if nme_count > 0 else 0
    nme_abs = nme_abs_batch_sum / nme_count if nme_count > 0 else 0
    nme_lms = nme_lms_batch_sum_total / nme_count if nme_count > 0 else np.zeros_like(nme_lms_batch_sum_total)
    nme_lms_abs = nme_lms_abs_batch_sum_total / nme_count if nme_count > 0 else np.zeros_like(nme_lms_abs_batch_sum_total)
    failure_001_rate = count_failure_001 / nme_count if nme_count > 0 else 0

    msg = 'Test Results time:{:.4f} nme:{:.6f} nme_abs:{:.6f} [001]:{:.6f}'.format(
        batch_time.avg, nme, nme_abs, failure_001_rate
    )

    logger.info(msg)

    return nme, nme_lms, predictions


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'predict')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}") 

    # Load the HRNet model structure
    pretrained_model = models.get_hrnet_wing_net(config).to(device)
    
    # --- FIXED: Load Trained Weights ---
    model_path = config.MODEL.PRETRAINED
    if os.path.isfile(model_path):
        logger.info(f"=> loading weights from checkpoint '{model_path}'")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: 
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # strict=True ensures the number of final output channels matches config.MODEL.NUM_JOINTS (12)
        pretrained_model.load_state_dict(new_state_dict, strict=True) 
        logger.info(f"=> loaded checkpoint '{model_path}'")
    else:
        logger.warning(f"=> no checkpoint found at '{model_path}'. Running with initialized weights!")
    # --- END FIX: Load Trained Weights ---
    
    # ALIGNMENT: Prepare the data with correct config handling for dataset.
    test_config_for_dataset = config.clone() 
    test_config_for_dataset.defrost() 
    test_config_for_dataset.DATASET.ROOT = config.DATASET.ROOT 
    test_config_for_dataset.freeze() 

    # Prepare the data
    test_data = WingLandmarksDataset(json_file=config.DATASET.TESTSET, cfg=test_config_for_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=config.TEST.BATCH_SIZE_PER_GPU, 
        shuffle=False, 
        num_workers=config.WORKERS,
        pin_memory=True 
    )

    # Run inference once over the entire test set
    nme, nme_lms, predictions = inference(config, test_loader, pretrained_model, device, logger)

    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))
    np.savetxt(os.path.join(final_output_dir, 'nme_on_landmarks.csv'), nme_lms, delimiter=',')


if __name__ == '__main__':
    main()
