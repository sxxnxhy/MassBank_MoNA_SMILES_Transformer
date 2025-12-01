import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
import logging
import sys

# Import your existing modules
import config
from dataset import prepare_dataloaders
from model import CLIPModel
# from model_baseline import BaselineCLIPModel as CLIPModel

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

K_CANDIDATES = 26159

@torch.no_grad()
def validate_benchmark_subsampled(model, val_loader, device, k_candidates=K_CANDIDATES):
    """
    Evaluates R@1 using the 'MassSpecGym / MIST' protocol.
    For each test spectrum, we rank the correct answer against (k-1) random decoys.
    """
    model.eval()
    print(f"\n" + "="*60)
    print(f"🔬 Running Benchmark Evaluation (Pool Size: {k_candidates})")
    print("="*60)
    
    all_spec_embeds = []
    all_text_embeds = []
    
    # 1. Extract Embeddings
    print("Encoding test set spectra and molecules...")
    for batch in tqdm(val_loader, desc="Encoding"):
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            spec_embeds = model.ms_encoder(peak_sequence, peak_mask)
            text_embeds = model.text_encoder(input_ids, attention_mask)
        
        # Move to CPU immediately to prevent OOM on large test sets
        all_spec_embeds.append(spec_embeds.cpu()) 
        all_text_embeds.append(text_embeds.cpu())
    
    all_spec_embeds = torch.cat(all_spec_embeds, dim=0).float()
    all_text_embeds = torch.cat(all_text_embeds, dim=0).float()
    
    num_samples = all_spec_embeds.shape[0]
    print(f"\n✅ Encoded {num_samples} test pairs.")
    
    # 2. Compute Full Similarity Matrix (N x N)
    # Optimization: If N is huge (>20k), we might need to chunk this. 
    # For ~13k, it fits in RAM fine.
    print("Computing similarity matrix...")
    sim_matrix = all_spec_embeds @ all_text_embeds.T
    
    hits = 0
    total = 0
    
    # 3. The "Benchmark" Loop
    print(f"Ranking each true answer against {k_candidates-1} random decoys...")
    
    # Set seed for reproducibility of the random sampling
    torch.manual_seed(config.RANDOM_SEED)
    
    for i in tqdm(range(num_samples), desc="Ranking"):
        # A. Get the score of the TRUE match (Diagonal)
        correct_score = sim_matrix[i, i]
        
        # B. Get scores of all FALSE matches (Everything else in the row)
        # We explicitly exclude index 'i' to ensure we don't compare against self
        neg_indices = torch.arange(num_samples) != i
        negative_scores = sim_matrix[i, neg_indices]
        
        # C. Randomly sample (k-1) negative scores
        # If we don't have enough negatives (rare), take all of them
        n_neg = min(len(negative_scores), k_candidates - 1)
        
        # Random permutation to select decoys
        perm = torch.randperm(len(negative_scores))[:n_neg]
        sampled_negatives = negative_scores[perm]
        
        # D. Compare: Rank 1 means Correct > ALL Decoys
        if correct_score >= sampled_negatives.max():
            hits += 1
        
        total += 1
        
    r1_benchmark = hits / total
    
    print("\n" + "="*60)
    print(f"🏆 FINAL RESULT")
    print(f"Benchmark R@1 (vs {k_candidates} candidates): {r1_benchmark*100:.2f}%")
    print("="*60)
    
    return r1_benchmark

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # 1. Load Data (Only need Test Loader)
    print("\n[1/3] Preparing Data...")
    # We ignore train_loader here
    _, test_loader = prepare_dataloaders()
    
    if test_loader is None:
        print("Error loading data.")
        return

    # 2. Load Model Architecture
    print("\n[2/3] Initializing Model...")
    model = CLIPModel().to(device)
    
    # 3. Load Checkpoint
    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.pt"
    print(f"\n[3/3] Loading Checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Successfully loaded model from Epoch {checkpoint['epoch']}")
        print(f"   (Previous Hard R@1: {checkpoint['val_metrics']['R@1']:.4f})")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("   Please make sure you ran training and have a saved model.")
        return

    # 4. Run Evaluation
    validate_benchmark_subsampled(model, test_loader, device, k_candidates=K_CANDIDATES)

if __name__ == '__main__':
    main()