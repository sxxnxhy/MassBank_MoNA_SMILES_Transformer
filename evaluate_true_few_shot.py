import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import config
from dataset import prepare_dataloaders
from model import CLIPModel
import torch.nn.functional as F

# --- [설정] ----------------------
K_SHOT_LIST = [1, 3, 5, 10]  # 힌트 개수
K_CANDIDATES = 256           # 경쟁자 수 (정답 1 + 오답 255)
# ---------------------------------

@torch.no_grad()
def evaluate_few_shot_benchmark(model, dataloader, device, k_shots=[1, 5], k_candidates=256):
    model.eval()
    print(f"\n" + "="*60)
    print(f"🔬 Running Few-Shot Benchmark Evaluation")
    print(f"   - Condition: {k_candidates} Candidates (1 True + {k_candidates-1} Decoys)")
    print(f"   - Logic: Average K spectra -> Rank against {k_candidates} candidates")
    print("="*60)

    # 1. 데이터 인코딩 및 그룹핑
    print("Encoding test set and grouping by SMILES...")
    
    mol_to_specs = defaultdict(list)
    mol_to_text_emb = {}
    
    # 배치 단위로 처리
    for batch in tqdm(dataloader, desc="Encoding"):
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 텍스트 디코딩
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
        smiles_list = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # 임베딩 추출
        with torch.cuda.amp.autocast():
            spec_emb = model.ms_encoder(peak_sequence, peak_mask)
            text_emb = model.text_encoder(input_ids, attention_mask)
            
        spec_emb = spec_emb.cpu()
        text_emb = text_emb.cpu()
        
        for i, smile in enumerate(smiles_list):
            smile_key = smile.replace(" ", "") # 공백 제거
            mol_to_specs[smile_key].append(spec_emb[i])
            if smile_key not in mol_to_text_emb:
                mol_to_text_emb[smile_key] = text_emb[i]

    # 2. 전체 후보군(Candidate Pool) 구축
    unique_smiles = list(mol_to_text_emb.keys())
    candidate_embeddings = torch.stack([mol_to_text_emb[s] for s in unique_smiles]) # [N_total, Dim]
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    
    print(f"\nTotal Unique Molecules in DB: {len(unique_smiles)}")
    
    # 3. K-Shot 별 성능 측정
    for k in k_shots:
        print(f"\n--- Testing {k}-Shot Retrieval (vs {k_candidates} candidates) ---")
        
        r1_hits = 0
        total_queries = 0
        
        # 랜덤 시드 고정 (재현성 위해)
        torch.manual_seed(config.RANDOM_SEED)
        
        for target_idx, target_smile in enumerate(tqdm(unique_smiles, desc=f"Evaluating (K={k})")):
            specs = mol_to_specs[target_smile]
            
            # 스펙트럼이 K개 미만이면 스킵
            if len(specs) < k:
                continue
                
            # [Step A] 힌트 생성 (K개 평균)
            indices = np.random.choice(len(specs), k, replace=False)
            selected_specs = torch.stack([specs[i] for i in indices])
            query_vec = torch.mean(selected_specs, dim=0, keepdim=True)
            query_vec = F.normalize(query_vec, p=2, dim=1)
            
            # [Step B] 전체 유사도 계산 (1 vs 4917)
            # 일단 전체랑 다 계산하고 나서, 나중에 256개만 추려내는 게 구현이 편함
            sim_scores = torch.matmul(query_vec, candidate_embeddings.T).squeeze() # [N_total]
            
            # [Step C] 256개 후보군 구성 (Subsampling)
            # 1. 정답 점수 확보
            score_target = sim_scores[target_idx]
            
            # 2. 오답 점수들만 모으기 (자신 제외)
            # 마스킹으로 자기 자신(target_idx)을 뺀 나머지 점수만 가져옴
            mask = torch.ones_like(sim_scores, dtype=torch.bool)
            mask[target_idx] = False
            negative_scores = sim_scores[mask]
            
            # 3. 랜덤으로 255개 오답 뽑기
            n_neg = min(len(negative_scores), k_candidates - 1)
            perm = torch.randperm(len(negative_scores))[:n_neg]
            sampled_negatives = negative_scores[perm]
            
            # [Step D] 랭킹 확인 (1 vs 256)
            # 내 점수가 뽑힌 오답들(255개) 중 가장 높은 점수보다 크면 1등
            if score_target > sampled_negatives.max():
                r1_hits += 1
                
            total_queries += 1
            
        # 결과 출력
        if total_queries > 0:
            print(f"  Samples evaluated: {total_queries}")
            print(f"  Benchmark R@1 : {r1_hits/total_queries*100:.2f}%")
        else:
            print("  No samples with enough spectra.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    _, test_loader = prepare_dataloaders()
    
    model = CLIPModel().to(device)
    checkpoint = torch.load(f"{config.CHECKPOINT_DIR}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model (Epoch {checkpoint['epoch']})")
    
    # 256개 후보군 설정
    evaluate_few_shot_benchmark(model, test_loader, device, k_shots=K_SHOT_LIST, k_candidates=256)

if __name__ == '__main__':
    main()