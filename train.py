"""
Training script for MassBank CLIP-style Zero-Shot Retrieval (ZSR).
(MODIFIED for "Path A": Transformer-on-Peaks)
"""
import os
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

import config
from dataset import prepare_dataloaders
from model import CLIPModel 

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@torch.no_grad()
def validate_zero_shot_retrieval(model, val_loader, device):
    """
    (MODIFIED) Passes peak_sequence and peak_mask to the model.
    """
    model.eval()
    print("Running Zero-Shot Retrieval (ZSR) evaluation...")
    
    all_spec_embeds = []
    all_text_embeds = []
    
    for batch in tqdm(val_loader, desc="Encoding ZSR test set"):
        # --- [NEW] ---
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        # --- [END NEW] ---
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with autocast('cuda'):
            spec_embeds = model.ms_encoder(peak_sequence, peak_mask) # <--- [MODIFIED]
            text_embeds = model.text_encoder(input_ids, attention_mask)
        
        all_spec_embeds.append(spec_embeds)
        all_text_embeds.append(text_embeds)
    
    all_spec_embeds = torch.cat(all_spec_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # ... (R@1 계산은 동일) ...
    num_test_samples = all_spec_embeds.shape[0]
    sim_matrix = all_spec_embeds @ all_text_embeds.T
    ground_truth = torch.arange(num_test_samples, device=device, dtype=torch.long)
    top10_indices = torch.topk(sim_matrix, k=10, dim=1)[1]
    top5_indices = top10_indices[:, :5]
    top1_indices = top10_indices[:, :1]
    acc_at_1 = (top1_indices == ground_truth.view(-1, 1)).any(dim=1).float().mean()
    acc_at_5 = (top5_indices == ground_truth.view(-1, 1)).any(dim=1).float().mean()
    acc_at_10 = (top10_indices == ground_truth.view(-1, 1)).any(dim=1).float().mean()
    metrics = { 'R@1': acc_at_1.item(), 'R@5': acc_at_5.item(), 'R@10': acc_at_10.item() }
    return metrics

def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch):
    """
    (MODIFIED) Passes peak_sequence and peak_mask to the model.
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # --- [NEW] ---
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        # --- [END NEW] ---
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            loss = model(peak_sequence, peak_mask, input_ids, attention_mask) # <--- [MODIFIED]
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def main():
    print("="*80)
    print("Training MassBank CLIP (v4 - Transformer-on-Peaks)")
    print("="*80)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR).mkdir(exist_ok=True)
    
    print("\nLoading data (Filtered: Centroid-Only)...")
    train_loader, val_loader = prepare_dataloaders()
    if train_loader is None:
        return
        
    print(f"Train batches: {len(train_loader)}, Test (ZSR) batches: {len(val_loader)}")
    
    print("\nCreating model...")
    model = CLIPModel().to(device)
    
    print(f"\nModel parameters (Trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    lora_params = model.text_encoder.bert.parameters()
    # ms_encoder.parameters()가 새 Transformer의 파라미터를 가져옴
    encoder_params = list(model.ms_encoder.parameters()) + \
                     list(model.text_encoder.projection.parameters()) + \
                     [model.logit_scale] 
    
    param_groups = [
        {'params': lora_params, 'lr': config.LR_LORA},
        {'params': encoder_params, 'lr': config.LR_ENCODER}
    ]

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.LR_LORA * 0.1 
    )
    
    scaler = GradScaler('cuda')
    writer = SummaryWriter(config.LOG_DIR)
    
    print(f"\nTraining for {config.NUM_EPOCHS} epochs...")
    print("="*80)
    
    best_r1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-"*80)
        start_time = time.time()
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        
        val_metrics = validate_zero_shot_retrieval(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.2f}s")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Val/R@1', val_metrics['R@1'], epoch)
        writer.add_scalar('Val/R@5', val_metrics['R@5'], epoch)
        writer.add_scalar('Val/R@10', val_metrics['R@10'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Val R@1: {val_metrics['R@1']:.4f}")
        print(f"  Val R@5: {val_metrics['R@5']:.4f}")
        print(f"  Val R@10: {val_metrics['R@10']:.4f}")
        
        if val_metrics['R@1'] > best_r1:
            best_r1 = val_metrics['R@1']
            best_epoch = epoch
            
            checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_metrics': val_metrics
            }, checkpoint_path)
            
            print(f"\n✓ Best model saved (R@1: {best_r1:.4f})")
    
    writer.close()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest R@1: {best_r1:.4f} (at epoch {best_epoch})")
    print(f"Model saved to: {config.CHECKPOINT_DIR}/best_model.pt")
    print(f"TensorBoard logs: {config.LOG_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()