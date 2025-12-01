import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import math

import config
from dataset import prepare_dataloaders
from model import CLIPModel

N_WAY = 5  #클래스 개수
K_SHOT = 5 #각 클래스당 참고할 샘플 수
N_QUERY = 5 #각 클래스당 맞춰야 할 문제 수
N_EPISODES = 600 #반복 횟수

@torch.no_grad()
def evaluate_few_shot_n_way(model, dataloader, device, n_way, k_shot, n_query, n_episodes):

    model.eval()
    print(f"{k_shot}-Shot {n_way}-Way {n_query}-Query {n_episodes}-Episode Evaluation")

    # 데이터를 클래스(SMILES)별로 정리 (Indexing)
    class_dict_spec = defaultdict(list) # Key: SMILES, Value: [Spectrum Embeddings]
    
    all_spec_embeds = []
    all_smiles = []
    
    for batch in tqdm(dataloader, desc="Encoding All Data"):
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        
        
        # 스펙트럼 임베딩 추출
        with torch.cuda.amp.autocast():
            spec_emb = model.ms_encoder(peak_sequence, peak_mask)
            
        spec_emb = spec_emb.cpu()
        
        # SMILES 텍스트 복원 (클래스 식별용)
        # dataset의 구조상 batch['input_ids']만 있으므로, 
        # 정확한 그룹핑을 위해 dataloader가 'smiles' 문자열도 반환하도록 dataset.py를 수정하면 좋지만,
        # 여기서는 편의상 unique한 input_ids 시퀀스를 key로 사용하겠습니다.
        input_ids = batch['input_ids'].cpu().numpy()
        
        for i in range(len(spec_emb)):
            # SMILES를 Key로 사용 (튜플로 변환하여 해시 가능하게 만듦)
            # 실제로는 SMILES 문자열이 더 좋지만, input_ids도 고유하므로 OK
            key = tuple(input_ids[i].tolist()) 
            class_dict_spec[key].append(spec_emb[i])

    # 샘플링 가능한 클래스 필터링
    # (K_shot + n_query)개 이상의 샘플을 가진 클래스만 사용 가능
    min_samples = k_shot + n_query
    valid_classes = [k for k, v in class_dict_spec.items() if len(v) >= min_samples]
    
    print(f"Total Classes: {len(class_dict_spec)}")
    print(f"Valid Classes (>= {min_samples} samples): {len(valid_classes)}")
    
    if len(valid_classes) < n_way:
        print(f" Error: Not enough valid classes. Need {n_way}, found {len(valid_classes)}.")
        return

    # 에피소드 반복 (Episodic Evaluation)
    accuracies = []
    
    for episode in tqdm(range(n_episodes), desc="Episodes"):
        # N개의 클래스 랜덤 선택
        chosen_keys = np.random.choice(len(valid_classes), n_way, replace=False)
        selected_classes = [valid_classes[k] for k in chosen_keys]
        
        support_set = [] # (N_way * K_shot, Dim)
        query_set = []   # (N_way * N_query, Dim)
        query_labels = []
        
        for label_idx, cls_key in enumerate(selected_classes):
            samples = class_dict_spec[cls_key]
            
            # 랜덤 샘플링 (비복원)
            indices = np.random.choice(len(samples), k_shot + n_query, replace=False)
            
            # Support Set (참고용 힌트)
            for k in indices[:k_shot]:
                support_set.append(samples[k])
            
            # Query Set (맞춰야 할 문제)
            for q in indices[k_shot:]:
                query_set.append(samples[q])
                query_labels.append(label_idx) # 0 ~ 4 사이의 정답 라벨
        
        # 텐서 변환
        support_set = torch.stack(support_set).to(device) # [25, 768]
        query_set = torch.stack(query_set).to(device)     # [75, 768]
        query_labels = torch.tensor(query_labels).to(device) # [75]
        
        # 프로토타입 계산 (Prototypical Networks 방식)
        # 각 클래스별(5개) Support Vector들의 평균을 구함
        # support_set을 [N_way, K_shot, Dim] 형태로 reshape
        prototypes = support_set.view(n_way, k_shot, -1).mean(dim=1) # [5, 768]
        
        # 거리(유사도) 계산 & 예측
        # Cosine Similarity 사용 
        # Query(N)와 Prototype(M) 간의 행렬 곱
        logits = torch.matmul(query_set, prototypes.T) # [75, 5]
        
        # 정확도 계산
        predictions = logits.argmax(dim=1)
        acc = (predictions == query_labels).float().mean().item()
        accuracies.append(acc)

    # 통계 계산
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    # 95% 신뢰구간 계산 (1.96 * std / sqrt(N))
    confidence = 1.96 * std_acc / math.sqrt(n_episodes)
    
    print(f"{k_shot}-Shot {n_way}-Way {n_query}-Query {n_episodes}-Episode")
    print(f"Avg Accuracy: {mean_acc:.2f}% , Std: {std_acc:.2f}%")

    
    return mean_acc, confidence

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
        print(f"Successfully loaded model from Epoch {checkpoint['epoch']}")
        print(f"(Previous Hard R@1: {checkpoint['val_metrics']['R@1']:.4f})")
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please make sure you ran training and have a saved model.")
        return

    # 4. Run Evaluation
    evaluate_few_shot_n_way(model, test_loader, device, n_way=N_WAY, k_shot=K_SHOT, n_query=N_QUERY, n_episodes=N_EPISODES)

if __name__ == '__main__':
    main()