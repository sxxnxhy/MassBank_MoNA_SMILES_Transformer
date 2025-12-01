import os
# Disable tokenizer parallelism to prevent deadlocks with DataLoader
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler 
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
import logging
import warnings

import config
from dataset import prepare_dataloaders
from model import CLIPModel 



@torch.no_grad()
def validate_zero_shot_retrieval(model, val_loader, device):
    model.eval()
    print("Running Zero-Shot Retrieval (ZSR) evaluation...")
    
    all_spec_embeds = []
    all_text_embeds = []
    
    for batch in tqdm(val_loader, desc="Encoding ZSR test set"):
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Use autocast for faster inference
        with autocast(device_type='cuda', dtype=torch.float16):
            spec_embeds = model.ms_encoder(peak_sequence, peak_mask)
            text_embeds = model.text_encoder(input_ids, attention_mask)
        
        all_spec_embeds.append(spec_embeds)
        all_text_embeds.append(text_embeds)
    
    all_spec_embeds = torch.cat(all_spec_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Calculate R@K
    num_test_samples = all_spec_embeds.shape[0]
    
    # Note: Similarity calculation is usually done in FP32 for precision
    sim_matrix = all_spec_embeds.float() @ all_text_embeds.float().T
    
    ground_truth = torch.arange(num_test_samples, device=device, dtype=torch.long)
    
    top10_indices = torch.topk(sim_matrix, k=10, dim=1)[1]
    top1_indices = top10_indices[:, :1]
    top5_indices = top10_indices[:, :5]
    
    acc_at_1 = (top1_indices == ground_truth.view(-1, 1)).any(dim=1).float().mean()
    acc_at_5 = (top5_indices == ground_truth.view(-1, 1)).any(dim=1).float().mean()
    acc_at_10 = (top10_indices == ground_truth.view(-1, 1)).any(dim=1).float().mean()
    
    metrics = { 'R@1': acc_at_1.item(), 'R@5': acc_at_5.item(), 'R@10': acc_at_10.item() }
    
    
    return metrics



if __name__ == "__main__":
    train_loader, val_loader = prepare_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel().to(device)
    
        # 2. Load Model Architecture
    print("\n[2/3] Initializing Model...")
    model = CLIPModel().to(device)
    
    # 3. Load Checkpoint
    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Successfully loaded model from Epoch {checkpoint['epoch']}")
    print(f"   (Previous Hard R@1: {checkpoint['val_metrics']['R@1']:.4f})")


    val_metrics = validate_zero_shot_retrieval(model, val_loader, device)
    print(f"  Val R@1: {val_metrics['R@1']:.4f}")
    print(f"  Val R@5: {val_metrics['R@5']:.4f}")
    print(f"  Val R@10: {val_metrics['R@10']:.4f}")
    