# ==============================================================================
# SCRIPT PURPOSE: Main training execution script for HRNet model using mixed-precision 
#                 (AMP), Adam optimizer, MultiStepLR scheduler, and validation checkpoints.
# INPUTS:         --cfg : Path to training YAML configuration file (e.g., hrnet_train.yaml)
# OUTPUTS:        Model checkpoints (epoch_X.pth, checkpoint.pth.tar) and TensorBoard logs
# PAPER SECTION:  Methods - Section 2.4 (HRNet Model Training Pipeline & Validation Loop)
# ==============================================================================

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models # This might not be directly used if models.py handles HRNet
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import pprint
import os
import shutil
import time # Added for timing validation epoch

import models # Assuming your HRNet model is defined in models.py
# --- CORRECTED CONFIG IMPORT ---
from config import update_config, _C as config # Import _C as config, as per your usage
# -------------------------------
from dataset import WingLandmarksDataset
# ensure these are from your utils.py which should have the corrected versions
from utils import get_preds, decode_preds, compute_nme
from utils import AverageMeter, create_logger 


# --- Diagnostic Print for utils.py path ---
import utils # This import is just for the diagnostic print below
print(f"train_hrnet.py is loading utils from: {utils.__file__}")
# ------------------------------------------


def save_checkpoint(state, is_best, checkpoint_dir='checkpoint'): 
    """Saves model and optimizer state during training.""" 
    os.makedirs(checkpoint_dir, exist_ok=True) 
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar') 
    torch.save(state, filename) 
    if is_best: 
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar')) 


def parse_args(): 
    parser = argparse.ArgumentParser(description='Train Face Alignment') 
    parser.add_argument('--cfg', help='experiment configuration filename', 
                        required=True, type=str) 
    args = parser.parse_args() 
    update_config(config, args) # The global 'config' object is updated here
    return args 


def validate(val_loader, model, criterion, epoch, writer_dict, device, logger): 
    num_classes = config.MODEL.NUM_JOINTS 
    losses = AverageMeter() 

    model.eval() 

    nme_count = 0 
    nme_batch_sum = 0 
    nme_abs_batch_sum = 0 
    count_failure_001 = 0 

    start_time = time.time() # Start timer for validation epoch

    with torch.no_grad(): 
        for i, (inp, target, meta) in enumerate(val_loader): 
            inp = inp.to(device) 
            
            # pts_gt comes from dataset's meta['pts'] which is (NUM_LANDMARKS, 3) (x, y, visibility)
            # We need (BATCH_SIZE, NUM_LANDMARKS, 2) for NME calculation.
            # meta['pts'] is already batched by the DataLoader, so it will be (BATCH_SIZE, NUM_LANDMARKS, 3)
            # Ensure it's on CPU and converted to numpy if needed for compute_nme.
            # The .cpu().numpy() here is correct for compute_nme.
            pts_gt = meta['pts'][:, :, 0:2].cpu().numpy() 

            # Updated to the non-deprecated version
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model(inp)

            score_map = output.data.cpu() # score_map is now a PyTorch Tensor on CPU

            target = target.to(device, non_blocking=True) 

            # loss 
            loss = criterion(output, target) 

            # --- CORRECTED CALL TO decode_preds ---
            # Pass score_map (PyTorch Tensor) directly to decode_preds.
            # meta['center'] and meta['scale'] are already PyTorch Tensors.
            preds = decode_preds(
                score_map,
                meta['center'],
                meta['scale'],
                config.MODEL.HEATMAP_SIZE 
            )
            # decode_preds will return a NumPy array, which is what compute_nme expects.
            # --- END CORRECTED CALL ---
            
            # NME 
            nme_temp, nme_abs_temp = compute_nme(preds, pts_gt) 
            
            # Failure Rate under different threshold 
            failure_001 = (nme_temp > 0.01).sum() 
            count_failure_001 += failure_001 

            nme_batch_sum += np.sum(nme_temp) 
            nme_abs_batch_sum += np.sum(nme_abs_temp) 
            nme_count = nme_count + preds.shape[0] # Use preds.shape[0] for actual batch size

            losses.update(loss.item(), inp.size(0)) 

    end_time = time.time() # End timer for validation epoch

    nme = nme_batch_sum / nme_count if nme_count > 0 else 0
    nme_abs = nme_abs_batch_sum / nme_count if nme_count > 0 else 0
    failure_001_rate = count_failure_001 / nme_count if nme_count > 0 else 0

    msg = 'Test Epoch {} loss:{:.6f} nme:{:.6f} nme_abs:{:.6f} [001]:{:.6f} Time: {:.2f}s'.format(
        epoch + 1, losses.avg, nme, nme_abs, failure_001_rate, end_time - start_time
    )
    logger.info(msg) 

    if writer_dict: 
        writer = writer_dict['writer'] 
        global_steps = writer_dict['valid_global_steps'] 
        writer.add_scalar('valid_loss', losses.avg, global_steps) 
        writer.add_scalar('valid_nme', nme, global_steps) 
        writer_dict['valid_global_steps'] = global_steps + 1 

    return nme 


def main(): 
    args = parse_args() 
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train') 

    logger.info(pprint.pformat(args)) 
    logger.info(pprint.pformat(config)) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    logger.info(f"Using device: {device}") 

    # Assuming models.get_hrnet_wing_net is the correct function to load your model
    pretrained_model = models.get_hrnet_wing_net(config).to(device) 

    #pretrained_model = torch.compile(pretrained_model)

    writer_dict = { 
        'writer': SummaryWriter(log_dir=tb_log_dir), 
        'train_global_steps': 0, 
        'valid_global_steps': 0, 
    } 

    # Your freezing logic 
    for param in pretrained_model.parameters(): 
        param.requires_grad = False 
    for param in pretrained_model.head.parameters(): 
        param.requires_grad = True 
    for param in pretrained_model.stage4.parameters(): 
        param.requires_grad = True 
    for param in pretrained_model.stage3.parameters(): 
        param.requires_grad = True 
    for param in pretrained_model.stage2.parameters(): 
        param.requires_grad = True 

    # Prepare the data 

    # --- For Training Dataset --- 
    # Clone the original config (which is already frozen by update_config) 
    train_config_for_dataset = config.clone() 
    # Defrost the cloned config before modifying it 
    train_config_for_dataset.defrost() 
    # Explicitly set the ROOT for the training dataset. 
    # This will use the value already loaded from DATASET.ROOT in your YAML. 
    train_config_for_dataset.DATASET.ROOT = config.DATASET.ROOT 
    # Freeze it again if you want this specific config for the dataset to be immutable 
    train_config_for_dataset.freeze() 

    train_data = WingLandmarksDataset(json_file=config.DATASET.TRAINSET, cfg=train_config_for_dataset) 
    train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, 
                                                shuffle=True, 
                                                num_workers=config.WORKERS, 
                                                pin_memory=True) 

    # --- For Validation Dataset --- 
    val_config_for_dataset = config.clone() 
    # Defrost the cloned config before modifying it 
    val_config_for_dataset.defrost() 
    # Set the ROOT specifically for the validation dataset using VAL_IMAGE_ROOT from YAML 
    val_config_for_dataset.DATASET.ROOT = config.DATASET.VAL_IMAGE_ROOT 
    # Freeze it again if you want this specific config for the dataset to be immutable 
    val_config_for_dataset.freeze() 

    val_data = WingLandmarksDataset(json_file=config.DATASET.VALSET, cfg=val_config_for_dataset) 
    val_loader = torch.utils.data.DataLoader(val_data, 
                                              batch_size=config.VAL.BATCH_SIZE_PER_GPU, 
                                              shuffle=False, # Standard practice for validation 
                                              num_workers=config.WORKERS, 
                                              pin_memory=True) 

    # Define the loss function and optimizer 
    criterion = nn.MSELoss().to(device) 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr=config.TRAIN.LR * config.TRAIN.WARMUP_RATIO) 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR) 

    # Initialize GradScaler for AMP 
    scaler = torch.amp.GradScaler(enabled=True) 

    # Train the model 
    num_epochs = config.TRAIN.END_EPOCH 
    best_nme = 1000 
    step = 0 
    for epoch in range(num_epochs): 
        pretrained_model.train() 
        # ... [training loop remains the same] ...
            step = step + 1 

        # evaluate 
        nme = validate(val_loader, pretrained_model, criterion, epoch, 
                                        writer_dict, device, logger) 
        is_best = nme < best_nme 
        best_nme = min(nme, best_nme) 
        scheduler.step() 

        # FIX: Call save_checkpoint here with the correct `is_best` flag
        save_checkpoint({ 
            'epoch': epoch + 1, 
            'model': pretrained_model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'best_nme': best_nme 
        }, is_best, checkpoint_dir=final_output_dir)

        # Save the trained model (epoch-specific)
        checkpoint_path = os.path.join(final_output_dir, 'epoch_{}.pth'.format(epoch+1)) 
        torch.save(pretrained_model.state_dict(), checkpoint_path)
        for i, (inputs, targets, _) in enumerate(train_loader): 
            # warmup 
            if step < config.TRAIN.WARMUP_ITERS: 
                lr = config.TRAIN.LR * (config.TRAIN.WARMUP_RATIO + (1 - config.TRAIN.WARMUP_RATIO) * (step + 1) / config.TRAIN.WARMUP_ITERS) 
                for param_group in optimizer.param_groups: 
                    param_group['lr'] = lr 

            inputs = inputs.to(device) 
            targets = targets.to(device) 

            optimizer.zero_grad() 

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16): 
                outputs = pretrained_model(inputs) 
                loss = criterion(outputs, targets) 

            scaler.scale(loss).backward() 
            scaler.step(optimizer) 
            scaler.update() 

            # Logging the loss for each batch 
            logger.info('[Step {}][Epoch {}/{}][Batch {}/{}] lr: {:.6f}, Loss: {:.6f}'.format(step+1, epoch+1, num_epochs, i+1, len(train_loader), optimizer.param_groups[0]['lr'], loss.item())) 
            writer_dict['train_global_steps'] += 1 
            writer_dict['writer'].add_scalar('train_loss', loss.item(), writer_dict['train_global_steps']) 
            writer_dict['writer'].add_scalar('train_lr', optimizer.param_groups[0]['lr'], writer_dict['train_global_steps']) 

            step = step + 1 

        # evaluate 
        nme = validate(val_loader, pretrained_model, criterion, epoch, 
                                        writer_dict, device, logger) 
        is_best = nme < best_nme 
        best_nme = min(nme, best_nme) 
        scheduler.step() 

        # Save the trained model 
        # The save_checkpoint function already handles saving best.pth.tar.
        # This part here saves epoch-specific model states.
        checkpoint_path = os.path.join(final_output_dir, 'epoch_{}.pth'.format(epoch+1)) 
        torch.save(pretrained_model.state_dict(), checkpoint_path) 
        
        # This symlink logic is redundant if save_checkpoint already copies 'model_best.pth.tar'
        # based on 'is_best'. If you want separate epoch.pth and a symlinked best.pth, then
        # you need to ensure save_checkpoint is saving to a .pth.tar and this is creating a .pth.
        # For consistency, I'd suggest modifying save_checkpoint to save `epoch_{}.pth` and `best.pth`
        # if `is_best` is true, or let `save_checkpoint` just handle `checkpoint.pth.tar` and `model_best.pth.tar`.
        # The current `save_checkpoint` creates `checkpoint.pth.tar` and `model_best.pth.tar`.
        # If you want `best.pth` as a symlink to `epoch_X.pth` for the best model,
        # then this part is correct, but `save_checkpoint` should only save `checkpoint.pth.tar`.
        # Let's assume you want both behaviors: generic checkpoint.pth.tar + model_best.pth.tar
        # AND epoch_X.pth + symlinked best.pth.
        # In that case, `save_checkpoint` needs to be used only for `checkpoint.pth.tar` and
        # `model_best.pth.tar`, and the code below for `epoch_X.pth` and `best.pth` is separate.
        
        # if is_best: 
        #     best_checkpoint_path = os.path.join(final_output_dir, 'best.pth')
        #     if os.path.exists(best_checkpoint_path): 
        #         os.remove(best_checkpoint_path) 
        #     checkpoint_abspath = os.path.abspath(checkpoint_path) 
        #     os.symlink(checkpoint_abspath, best_checkpoint_path) 


if __name__ == '__main__': 
    main()
