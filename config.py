
# Data Source
MASSBANK_FILE = "./data_preprocess/massbank_300peaks.parquet"
# MASSBANK_FILE = "./data_preprocess/massbank_num_peak_limit_30.parquet"

MONA_FILE = "./data_preprocess/mona_30peaks_1064635.parquet"



# ------------------------------------------------------------------------------
# Logging
CHECKPOINT_DIR = './models_massbank_v4_PeakTransformer'
LOG_DIR = './logs_massbank_v4_PeakTransformer'
RANDOM_SPLIT = False
# Device
DEVICE = 'cuda:0'
# ------------------------------------------------------------------------------
# # MoNA 추가 버전 설정 
# # Logging
# CHECKPOINT_DIR = './models_massbank_v4_PeakTransformer_mona_added'
# LOG_DIR = './logs_massbank_v4_PeakTransformer_mona_added'
# RANDOM_SPLIT = True
# # Device
# DEVICE = 'cuda:0'
# ------------------------------------------------------------------------------
# # baseline 설정
# # Logging
# CHECKPOINT_DIR = './models_massbank_baseline'
# LOG_DIR = './logs_massbank_baseline'
# RANDOM_SPLIT = False
# # Device
# DEVICE = 'cuda:0'
# ------------------------------------------------------------------------------
# Zero-Shot Setup
TRAIN_TEST_SPLIT_RATIO = 0.8  
RANDOM_SEED = 42

MAX_PEAK_SEQ_LEN = 300    # 스펙트럼 당 최대 피크 (토큰) 시퀀스 길이

# Model Architecture
EMBEDDING_DIM = 768       # 공유 임베딩 차원 (BERT와 일치)


MS_ENCODER = {
    # Gaussian Fourier Projection Dimension (Internal)
    # This projects scalar m/z -> vector of size 'fourier_dim'
    'fourier_dim': 256,
    # (m/z, intensity) 2D 입력을 d_model로 임베딩
    'd_model': EMBEDDING_DIM, 
    'nhead': 8,           # (d_model % nhead == 0)
    'n_layers': 6,        # 트랜스포머 레이어 수
    'dropout': 0.1
}

# loss temperature
TEMPERATURE = 0.07

# Text Encoder
TEXT_ENCODER = {
    'model_name': 'seyonec/PubChem10M_SMILES_BPE_450k',
    'max_length': 365,  
    'freeze_bert': False
}

LORA = {
    'r': 16,  # Rank (hyperparameter, 8 or 16 is common)
    'lora_alpha' : 32, # (hyperparameter, often 2*r)
    'lora_dropout': 0.1,
    'target_modules': ["query", "key", "value"] # Apply to attention layers
}

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 500 
WEIGHT_DECAY = 1e-2 

# Learning Rates 
LR_LORA = 1e-4
LR_ENCODER = 1e-4 





