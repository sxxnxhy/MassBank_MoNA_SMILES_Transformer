"""
Configuration for MassBank Zero-Shot Retrieval (ZSR)
(MODIFIED for "Path A": Transformer-on-Peaks)
"""

# --- Data Source ---
MASSBANK_FILE = "./data_preprocess/massbank_30peaks_98675.parquet"
MONA_FILE = "./data_preprocess/mona_30peaks_1064635.parquet"

# --- Zero-Shot Setup ---
TRAIN_TEST_SPLIT_RATIO = 0.8  
RANDOM_SEED = 42

# (이전 로그 99백분위수 655, 필터 1000을 기준으로 설정)
MAX_PEAK_SEQ_LEN = 30    # 스펙트럼 당 최대 피크 (토큰) 시퀀스 길이

# --- Model Architecture ---
EMBEDDING_DIM = 768       # 공유 임베딩 차원 (BERT와 일치)

# --- (NEW) MS Encoder (Transformer-on-Peaks) ---
MS_ENCODER = {
    # (m/z, intensity) 2D 입력을 d_model로 임베딩
    'd_model': EMBEDDING_DIM, 
    'nhead': 8,           # (d_model % nhead == 0)
    'n_layers': 6,        # 트랜스포머 레이어 수
    'dropout': 0.1
}

# loss temperature
TEMPERATURE = 0.07

# Text Encoder (ChemBERTa)
TEXT_ENCODER = {
    'model_name': 'seyonec/PubChem10M_SMILES_BPE_450k',
    'max_length': 369,  
    'freeze_bert': False
}

LORA = {
    'r': 16,  # Rank (hyperparameter, 8 or 16 is common)
    'lora_alpha' : 32, # (hyperparameter, often 2*r)
    'lora_dropout': 0.1,
    'target_modules': ["query", "key", "value"] # Apply to attention layers
}

# Training
BATCH_SIZE = 256
NUM_EPOCHS = 500 
WEIGHT_DECAY = 1e-2 

# Learning Rates 
LR_LORA = 1e-4
LR_ENCODER = 1e-4 

# Logging
CHECKPOINT_DIR = './models_massbank_v4_PeakTransformer'
LOG_DIR = './logs_massbank_v4_PeakTransformer'

# Device
DEVICE = 'cuda:0'


