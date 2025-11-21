import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import config
from dataset import prepare_dataloaders
from model import CLIPModel
import torch.nn.functional as F

# --- [설정] ----------------------
K_SHOT_LIST = [1, 3, 5, 10]  # 테스트할 샷 수 (1장, 3장, 5장, 10장 줬을 때 성능 변화)
# ---------------------------------

@torch.no_grad()
def evaluate_few_shot_retrieval_scan(model, dataloader, device, k_shots=[1, 5]):
    model.eval()
    print(f"\n" + "="*60)
    print(f"🔎 Running Few-Shot Retrieval Benchmark (K={k_shots})")
    print("Logic: Average K spectra -> Retrieve correct SMILES from FULL database")
    print("="*60)

    # 1. 데이터 인코딩 및 그룹핑
    print("Encoding test set and grouping by SMILES...")
    
    # Key: SMILES string, Value: List of Spectrum Embeddings
    mol_to_specs = defaultdict(list)
    # Key: SMILES string, Value: Text Embedding (1개만 있으면 됨)
    mol_to_text_emb = {}
    
    # 배치 단위로 처리
    for batch in tqdm(dataloader, desc="Encoding"):
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 텍스트 디코딩 (그룹핑 키로 사용)
        # 주의: 실제 dataset.py에 get_tokenizer가 있어야 함. 없으면 config에서 로드.
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
            # 공백 제거 (토크나이저 디코딩 시 생길 수 있는 공백 처리)
            smile_key = smile.replace(" ", "") 
            
            mol_to_specs[smile_key].append(spec_emb[i])
            if smile_key not in mol_to_text_emb:
                mol_to_text_emb[smile_key] = text_emb[i]

    # 2. 검색 대상(Candidate Pool) 구축
    # 전체 유니크한 SMILES들의 텍스트 임베딩 행렬
    unique_smiles = list(mol_to_text_emb.keys())
    candidate_embeddings = torch.stack([mol_to_text_emb[s] for s in unique_smiles]) # [N_unique, Dim]
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1) # 정규화
    
    print(f"\nCandidate Pool Size (Unique Molecules): {len(unique_smiles)}")
    
    # 3. K-Shot 별 성능 측정
    for k in k_shots:
        print(f"\n--- Testing {k}-Shot Retrieval ---")
        
        r1_hits = 0
        r5_hits = 0
        r10_hits = 0
        total_queries = 0
        
        # 각 분자마다 루프
        for target_smile in tqdm(unique_smiles, desc=f"Retrieving (K={k})"):
            specs = mol_to_specs[target_smile]
            
            # 스펙트럼 개수가 K개 미만이면 테스트 불가 (스킵)
            if len(specs) < k:
                continue
                
            # K개 랜덤 샘플링 (비복원) -> 평균 벡터 생성
            # 실험의 안정성을 위해, 가능한 경우 여러 번 샘플링해서 평균낼 수도 있지만
            # 여기서는 1번만 수행 (Standard Protocol)
            indices = np.random.choice(len(specs), k, replace=False)
            selected_specs = torch.stack([specs[i] for i in indices]) # [K, Dim]
            
            # [핵심] Mean Pooling (벡터 평균)
            query_vec = torch.mean(selected_specs, dim=0, keepdim=True) # [1, Dim]
            query_vec = F.normalize(query_vec, p=2, dim=1)
            
            # 유사도 계산 (1 vs N)
            sim_scores = torch.matmul(query_vec, candidate_embeddings.T).squeeze() # [N_unique]
            
            # 랭킹 계산
            # 정답 인덱스 찾기
            target_idx = unique_smiles.index(target_smile)
            
            # 내림차순 정렬 후 정답 등수 확인
            # (argsort는 오름차순이므로 뒤집거나, '보다 큰 값의 개수'를 셈)
            score_target = sim_scores[target_idx]
            rank = (sim_scores > score_target).sum().item() + 1
            
            if rank == 1: r1_hits += 1
            if rank <= 5: r5_hits += 1
            if rank <= 10: r10_hits += 1
            total_queries += 1
            
        # 결과 출력
        if total_queries == 0:
            print("  Warning: No molecules had enough spectra for this K.")
        else:
            print(f"  Samples evaluated: {total_queries}")
            print(f"  R@1 : {r1_hits/total_queries*100:.2f}%")
            print(f"  R@5 : {r5_hits/total_queries*100:.2f}%")
            print(f"  R@10: {r10_hits/total_queries*100:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # 데이터 로드
    _, test_loader = prepare_dataloaders()
    
    # 모델 로드
    model = CLIPModel().to(device)
    checkpoint = torch.load(f"{config.CHECKPOINT_DIR}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model (Epoch {checkpoint['epoch']})")
    
    # 평가 실행
    evaluate_few_shot_retrieval_scan(model, test_loader, device, k_shots=K_SHOT_LIST)

if __name__ == '__main__':
    main()