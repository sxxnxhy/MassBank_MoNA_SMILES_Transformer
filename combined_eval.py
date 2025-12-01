import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import math
import torch.nn.functional as F
from torch.amp import autocast

import config
from dataset import prepare_dataloaders
from model import CLIPModel

# --- [설정] ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = f"{config.CHECKPOINT_DIR}/best_model.pt"

# Few-Shot 설정
N_WAY = 5
N_QUERY = 5
N_EPISODES = 600

def load_model_and_data():
    """모델과 데이터 로더를 준비합니다."""
    print("\n[1/4] Loading Data & Model...")
    
    # 1. 데이터 로드 (Test Loader만 사용)
    _, val_loader = prepare_dataloaders()
    
    # 2. 모델 초기화
    model = CLIPModel().to(DEVICE)
    
    # 3. 체크포인트 로드
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading Checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from Epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        
    return model, val_loader

@torch.no_grad()
def encode_all_data(model, val_loader):
    """
    테스트 셋 전체를 한 번만 인코딩하여 GPU/CPU 메모리에 저장합니다.
    모든 실험에서 이 임베딩을 재사용합니다.
    """
    model.eval()
    print("\n[2/4] Encoding All Test Data (Running Inference Once)...")
    
    all_spec_embeds = []
    all_text_embeds = []
    all_input_ids = [] # 클래스 식별용 (SMILES 대용)
    
    for batch in tqdm(val_loader, desc="Encoding"):
        peak_sequence = batch['peak_sequence'].to(DEVICE).float()
        peak_mask = batch['peak_mask'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            spec_emb = model.ms_encoder(peak_sequence, peak_mask)
            text_emb = model.text_encoder(input_ids, attention_mask)
        
        # CPU로 이동하여 저장
        all_spec_embeds.append(spec_emb.cpu())
        all_text_embeds.append(text_emb.cpu())
        all_input_ids.append(input_ids.cpu())
        
    # 텐서 합치기
    all_spec_embeds = torch.cat(all_spec_embeds, dim=0).float()
    all_text_embeds = torch.cat(all_text_embeds, dim=0).float()
    all_input_ids = torch.cat(all_input_ids, dim=0)
    
    print(f"✅ Encoded {all_spec_embeds.shape[0]} samples.")
    return all_spec_embeds, all_text_embeds, all_input_ids

# ---------------------------------------------------------
# 실험 1 & 2: Zero-Shot Retrieval (Global vs Benchmark)
# ---------------------------------------------------------
def evaluate_retrieval(spec_embeds, text_embeds, k_candidates=None):
    """
    k_candidates=None -> Global (Hard)
    k_candidates=256 -> Benchmark (Standard)
    """
    num_samples = spec_embeds.shape[0]
    mode_name = f"Pool: {k_candidates}" if k_candidates else f"Pool: ALL ({num_samples})"
    print(f"\n   Running Retrieval Eval [{mode_name}]...")
    
    # 유사도 행렬 계산
    sim_matrix = spec_embeds.to(DEVICE) @ text_embeds.to(DEVICE).T
    
    hits_1, hits_5, hits_10 = 0, 0, 0
    
    # --- Case A: Global Retrieval (전체 검색) ---
    if k_candidates is None or k_candidates >= num_samples:
        ground_truth = torch.arange(num_samples, device=DEVICE, dtype=torch.long)
        _, top10_indices = torch.topk(sim_matrix, k=10, dim=1)
        
        hits_1 = (top10_indices[:, :1] == ground_truth.view(-1, 1)).any(dim=1).sum().item()
        hits_5 = (top10_indices[:, :5] == ground_truth.view(-1, 1)).any(dim=1).sum().item()
        hits_10 = (top10_indices[:, :10] == ground_truth.view(-1, 1)).any(dim=1).sum().item()
        
    # --- Case B: Benchmark Subsampling (256개 중 검색) ---
    else:
        torch.manual_seed(config.RANDOM_SEED) # 재현성
        
        for i in range(num_samples):
            correct_score = sim_matrix[i, i]
            
            # 나 자신 제외한 오답 점수들
            neg_indices = torch.arange(num_samples) != i
            negative_scores = sim_matrix[i, neg_indices]
            
            # 255개 랜덤 샘플링
            n_neg = k_candidates - 1
            perm = torch.randperm(len(negative_scores))[:n_neg]
            sampled_negatives = negative_scores[perm]
            
            # 랭킹 (점수가 더 높은 오답 개수 + 1)
            # 동점자 처리를 위해 > 사용 (일반적 기준)
            rank = (sampled_negatives > correct_score).sum().item() + 1
            
            if rank == 1: hits_1 += 1
            if rank <= 5: hits_5 += 1
            if rank <= 10: hits_10 += 1
            
    return {
        'R@1': hits_1 / num_samples * 100,
        'R@5': hits_5 / num_samples * 100,
        'R@10': hits_10 / num_samples * 100
    }

# ---------------------------------------------------------
# 실험 3 & 4: Few-Shot Classification (1-Shot & 5-Shot)
# ---------------------------------------------------------
def evaluate_few_shot(spec_embeds, input_ids, n_way=5, k_shot=1, n_query=5, n_episodes=600):
    print(f"\n   Running {k_shot}-Shot {n_way}-Way Classification...")
    
    # 1. 클래스별 인덱싱
    class_indices = defaultdict(list)
    for idx, token_ids in enumerate(input_ids):
        # Tensor -> Tuple (Hashable Key)
        key = tuple(token_ids.tolist())
        class_indices[key].append(idx)
        
    # 유효 클래스 필터링 (데이터 충분한 것만)
    min_samples = k_shot + n_query
    valid_classes = [k for k, v in class_indices.items() if len(v) >= min_samples]
    
    if len(valid_classes) < n_way:
        print(f"⚠️ Error: Not enough classes with {min_samples} samples.")
        return 0.0, 0.0

    accuracies = []
    
    # 2. 에피소드 반복
    for _ in range(n_episodes):
        # 클래스 랜덤 선택
        chosen_keys_idx = np.random.choice(len(valid_classes), n_way, replace=False)
        chosen_keys = [valid_classes[i] for i in chosen_keys_idx]
        
        support_set = []
        query_set = []
        query_labels = []
        
        for label_idx, key in enumerate(chosen_keys):
            indices = class_indices[key]
            selected_indices = np.random.choice(indices, k_shot + n_query, replace=False)
            
            # Support & Query 분리
            sup_idx = selected_indices[:k_shot]
            qry_idx = selected_indices[k_shot:]
            
            # 임베딩 가져오기 (GPU로 이동)
            # k_shot이 1일 때도 차원 유지를 위해 stack 사용
            sup_emb = torch.stack([spec_embeds[i] for i in sup_idx]).to(DEVICE)
            qry_emb = torch.stack([spec_embeds[i] for i in qry_idx]).to(DEVICE)
            
            # 프로토타입 (평균)
            prototype = sup_emb.mean(dim=0)
            support_set.append(prototype)
            
            query_set.append(qry_emb)
            query_labels.extend([label_idx] * n_query)
            
        # 텐서 변환 [N_way, Dim], [Total_Query, Dim]
        prototypes = torch.stack(support_set)
        queries = torch.cat(query_set)
        labels = torch.tensor(query_labels).to(DEVICE)
        
        # 거리 계산 & 예측
        logits = torch.matmul(queries, prototypes.T)
        preds = logits.argmax(dim=1)
        
        acc = (preds == labels).float().mean().item()
        accuracies.append(acc)
        
    mean = np.mean(accuracies) * 100
    std = np.std(accuracies) * 100
    return mean, std

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # 1. 모델 & 데이터 로드
    model, val_loader = load_model_and_data()
    
    # 2. 전체 데이터 인코딩 (한 번만 수행)
    spec_embeds, text_embeds, input_ids = encode_all_data(model, val_loader)
    
    print("\n" + "="*60)
    print("🚀 FINAL RESULTS SUMMARY")
    print("="*60)
    
    # --- [Result 1] Zero-Shot (Hard Mode: 1 vs All) ---
    res1 = evaluate_retrieval(spec_embeds, text_embeds, k_candidates=None)
    print(f"1. Zero-Shot Global Retrieval (Hard Mode, vs {spec_embeds.shape[0]})")
    print(f"   R@1: {res1['R@1']:.2f}% | R@10: {res1['R@10']:.2f}%")
    print("-" * 60)
    
    # --- [Result 2] Zero-Shot (Benchmark: 1 vs 256) ---
    res2 = evaluate_retrieval(spec_embeds, text_embeds, k_candidates=256)
    print(f"2. Zero-Shot Benchmark Retrieval (Standard, vs 256)")
    print(f"   R@1: {res2['R@1']:.2f}% | R@10: {res2['R@10']:.2f}%")
    print("-" * 60)
    
    # --- [Result 3] 1-Shot Classification ---
    acc_1shot, std_1shot = evaluate_few_shot(
        spec_embeds, input_ids, n_way=N_WAY, k_shot=1, n_query=N_QUERY, n_episodes=N_EPISODES
    )
    print(f"3. Few-Shot Classification (1-Shot, 5-Way)")
    print(f"   Accuracy: {acc_1shot:.2f}% ± {std_1shot:.2f}%")
    print("-" * 60)

    # --- [Result 4] 5-Shot Classification ---
    acc_5shot, std_5shot = evaluate_few_shot(
        spec_embeds, input_ids, n_way=N_WAY, k_shot=5, n_query=N_QUERY, n_episodes=N_EPISODES
    )
    print(f"4. Few-Shot Classification (5-Shot, 5-Way)")
    print(f"   Accuracy: {acc_5shot:.2f}% ± {std_5shot:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()