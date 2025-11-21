import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from transformers import AutoTokenizer
import config
from dataset import prepare_dataloaders
from model import CLIPModel

# =============================================================================
# 🏆 MASSSPECGYM / SOTA BENCHMARK PROTOCOL
# =============================================================================
# 1. Search Space: Entire Test Set (No subsampling)
# 2. Hit Criteria: Exact SMILES Match (Structure-Level)
# 3. Ranking: Optimistic Rank (Standard for retrieval with duplicates)
# =============================================================================

@torch.no_grad()
def evaluate_sota_benchmark(model, val_loader, device):
    model.eval()
    # Load tokenizer to decode SMILES
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    
    print(f"\n" + "="*60)
    print(f"🔬 Running MassSpecGym / SOTA Benchmark")
    print(f"   - Criteria: Structure (SMILES) Match")
    print(f"   - Search Space: Full Database (All Test Spectra)")
    print("="*60)
    
    all_spec_embeds = []
    all_text_embeds = []
    all_smiles = []
    
    # --- 1. Encode All Data & Decode SMILES ---
    print("Encoding test set and extracting SMILES...")
    for batch in tqdm(val_loader, desc="Encoding"):
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            spec_embeds = model.ms_encoder(peak_sequence, peak_mask)
            text_embeds = model.text_encoder(input_ids, attention_mask)
        
        all_spec_embeds.append(spec_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        
        # Decode SMILES for structure matching
        decoded_smiles = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # Remove spaces (tokenizers often add spaces)
        all_smiles.extend([s.replace(" ", "") for s in decoded_smiles])
    
    # Stack embeddings
    all_spec_embeds = torch.cat(all_spec_embeds, dim=0).float() # [N, Dim]
    all_text_embeds = torch.cat(all_text_embeds, dim=0).float() # [N, Dim]
    
    num_samples = len(all_smiles)
    all_smiles_arr = np.array(all_smiles)
    
    print(f"\n✅ Encoded {num_samples} spectra.")
    print(f"ℹ️  Unique Molecules in Test Set: {len(set(all_smiles))}")
    
    # --- 2. Compute Full Similarity Matrix ---
    print("Computing N x N similarity matrix...")
    # This fits in ~16GB RAM for 26k x 26k. If OOM, use chunking.
    sim_matrix = all_spec_embeds @ all_text_embeds.T
    
    # --- 3. Evaluate Ranking (The SOTA Logic) ---
    hits_r1 = 0
    hits_r5 = 0
    hits_r10 = 0
    
    print("Calculating R@K using Structure Matching...")
    for i in tqdm(range(num_samples), desc="Ranking"):
        query_smile = all_smiles[i]
        scores = sim_matrix[i] # [N]
        
        # A. Find all indices that are "Correct Answers" (Same SMILES)
        #    (Including the query itself, and all its duplicates)
        is_correct_structure = (all_smiles_arr == query_smile)
        
        # B. Find the Best Score among ALL correct answers
        #    (We assume the model succeeds if *any* instance of the molecule is top-ranked)
        correct_scores = scores[is_correct_structure]
        best_correct_score = correct_scores.max().item()
        
        # C. Count how many WRONG molecules have a higher score
        #    Strictly greater (>), so ties count as a win (Optimistic Rank).
        is_incorrect_structure = ~is_correct_structure
        incorrect_scores = scores[is_incorrect_structure]
        
        # Rank = 1 + (Number of distractors that beat our best correct score)
        num_better_distractors = (incorrect_scores > best_correct_score).sum().item()
        rank = 1 + num_better_distractors
        
        if rank <= 1: hits_r1 += 1
        if rank <= 5: hits_r5 += 1
        if rank <= 10: hits_r10 += 1
        
    r1 = hits_r1 / num_samples * 100
    r5 = hits_r5 / num_samples * 100
    r10 = hits_r10 / num_samples * 100
    
    print("\n" + "="*60)
    print(f"🏆 SOTA COMPARISON RESULTS")
    print("-" * 30)
    print(f"   Pool Size: {num_samples} (Full Test Set)")
    print(f"   Metric   : Structure Retrieval Accuracy (SMILES Match)")
    print("-" * 30)
    print(f"   R@1  : {r1:.2f}%")
    print(f"   R@5  : {r5:.2f}%")
    print(f"   R@10 : {r10:.2f}%")
    print("="*60)

def main():
    # Check device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Load Data
    _, test_loader = prepare_dataloaders()
    
    # Load Model
    model = CLIPModel().to(device)
    checkpoint = torch.load(f"{config.CHECKPOINT_DIR}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded Model Epoch: {checkpoint['epoch']}")
    
    # Run Eval
    evaluate_sota_benchmark(model, test_loader, device)

if __name__ == '__main__':
    main()