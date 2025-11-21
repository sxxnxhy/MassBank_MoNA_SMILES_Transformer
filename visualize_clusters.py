import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
from collections import defaultdict, Counter
import pandas as pd
from sklearn.decomposition import PCA
# UMAP은 별도 설치가 필요할 수 있음 (pip install umap-learn)
# 없으면 t-SNE만 작동하도록 예외 처리
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("! UMAP library not found. Will skip UMAP and run t-SNE only.")
    print("  (To use UMAP: pip install umap-learn)")

import config
from dataset import prepare_dataloaders
from model import CLIPModel
from transformers import AutoTokenizer

# --- [설정] ---
TOP_K_MOLECULES = 20   # 시각화할 분자 종류 수 (너무 많으면 색깔 구분이 안 됨)
SCATTER_POINT_SIZE = 50
FIG_SIZE = (16, 8)
# -------------

@torch.no_grad()
def visualize_embedding_space(model, dataloader, device):
    model.eval()
    print("\n" + "="*60)
    print("🎨 Generating Embedding Space Visualization (t-SNE / UMAP)")
    print("="*60)

    # 1. 데이터 인코딩 (전체 테스트 셋)
    print("Encoding test set...")
    
    all_spec_embeds = []
    all_smiles = []
    
    # 토크나이저 로드 (SMILES 복원용)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    
    for batch in tqdm(dataloader, desc="Encoding"):
        peak_sequence = batch['peak_sequence'].to(device).float()
        peak_mask = batch['peak_mask'].to(device)
        input_ids = batch['input_ids']
        
        # 임베딩 추출
        with torch.cuda.amp.autocast():
            spec_emb = model.ms_encoder(peak_sequence, peak_mask)
            
        # CPU로 이동
        all_spec_embeds.append(spec_emb.cpu().numpy())
        
        # SMILES 디코딩
        decoded_smiles = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # 공백 제거
        decoded_smiles = [s.replace(" ", "") for s in decoded_smiles]
        all_smiles.extend(decoded_smiles)

    all_spec_embeds = np.concatenate(all_spec_embeds, axis=0)
    all_smiles = np.array(all_smiles)
    
    print(f"Total Spectra Encoded: {len(all_spec_embeds)}")
    print(f"Total Unique Molecules: {len(set(all_smiles))}")

    # 2. 상위 K개 분자 필터링 (데이터 많은 순서대로)
    print(f"\nSelecting Top {TOP_K_MOLECULES} most frequent molecules for visualization...")
    
    counter = Counter(all_smiles)
    top_molecules = [m for m, c in counter.most_common(TOP_K_MOLECULES)]
    
    # 필터링된 데이터 인덱스 찾기
    mask = np.isin(all_smiles, top_molecules)
    filtered_embeds = all_spec_embeds[mask]
    filtered_labels = all_smiles[mask]
    
    print(f"Selected {len(filtered_embeds)} spectra from {TOP_K_MOLECULES} molecules.")

    # 3. t-SNE 실행
    print("\nRunning t-SNE (This may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
    tsne_result = tsne.fit_transform(filtered_embeds)
    
    # 4. 시각화 (t-SNE)
    plot_scatter(tsne_result, filtered_labels, "t-SNE Visualization of Physics-Informed Embeddings", "tsne_plot.png")

    # 5. UMAP 실행 (설치된 경우)
    if HAS_UMAP:
        print("\nRunning UMAP...")
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(filtered_embeds)
        plot_scatter(umap_result, filtered_labels, "UMAP Visualization of Physics-Informed Embeddings", "umap_plot.png")
        
    # --- [추가] 6. PCA 실행 ---
    print("\nRunning PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(filtered_embeds)
    
    # 분산 설명력(Explained Variance) 출력 (이게 높을수록 PCA가 믿을만함)
    exp_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA Explained Variance Ratio (2D): {exp_var:.2f}%")
    
    plot_scatter(
        pca_result, 
        filtered_labels, 
        f"PCA Visualization (Explained Variance: {exp_var:.1f}%)", 
        "pca_plot.png"
    )

def plot_scatter(points, labels, title, filename):
    """예쁜 산점도를 그리는 함수"""
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'Molecule': labels
    })
    
    plt.figure(figsize=FIG_SIZE)
    sns.set_style("whitegrid")
    
    # Scatter Plot
    sns.scatterplot(
        data=df, x='x', y='y', hue='Molecule', 
        palette=sns.color_palette("hls", len(set(labels))),
        s=SCATTER_POINT_SIZE, alpha=0.8, edgecolor='k'
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="SMILES")
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved plot to {filename}")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # 데이터 로드
    _, test_loader = prepare_dataloaders()
    
    # 모델 로드
    model = CLIPModel().to(device)
    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.pt"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from Epoch {checkpoint['epoch']}")
    except FileNotFoundError:
        print("Checkpoint not found!")
        return

    # 시각화 실행
    visualize_embedding_space(model, test_loader, device)

if __name__ == '__main__':
    main()