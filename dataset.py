import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
import config
from tqdm import tqdm

def get_tokenizer():
    """Initializes and returns the tokenizer."""
    return AutoTokenizer.from_pretrained(
            config.TEXT_ENCODER['model_name']
        )

class MassBankDataset(Dataset):
    """
    (CORRECTED "Path A")
    - m/z: 로그 변환 (log(mz))
    - intensity: 로컬 정규화 (norm_int)
    - 버그 수정 (np.ndarray -> list)
    """
    def __init__(self, df, tokenizer, is_train=False): 
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train
        
        # --- (REMOVED) m/z min/max 정규화 삭제 ---
        
        self.MAX_LEN = config.MAX_PEAK_SEQ_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. Text (SMILES) ---
        smiles = row.get('smiles')
        text_prompt = f"{smiles}"
        
        tokenized_text = self.tokenizer(
            text_prompt,
            max_length=config.TEXT_ENCODER['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # --- 2. Spectrum (Peak Sequence) ---
        peak_list = row['peaks'] 
        
        # [BUG FIX 1] Parquet의 ndarray(object)를 Python list로 변환
        if isinstance(peak_list, np.ndarray):
            peak_list = peak_list.tolist()

        processed_peaks = []
        
        # --- [Intensity 정규화 (Local)] ---
        if peak_list:
            intensities = [p[1] for p in peak_list]
            max_int = max(intensities)
            if max_int < 1e-10: max_int = 1.0 # 0으로 나누기 방지
        else:
            max_int = 1.0

        for mz, intensity in peak_list:
            # --- [m/z 변환: Log] ---
            # 0 또는 음수 m/z 방지 (log(1) = 0)
            log_mz = np.log(mz + 1.0) 
            
            # --- [Intensity 정규화 (Local)] ---
            norm_int = intensity / max_int
            
            processed_peaks.append((log_mz, norm_int))
            
        # --- 3. Padding / Truncation ---
        num_peaks = len(processed_peaks)
        
        peak_sequence = torch.zeros(self.MAX_LEN, 2, dtype=torch.float32)
        peak_mask = torch.zeros(self.MAX_LEN, dtype=torch.bool)

        if num_peaks > 0:
            # [BUG FIX 2] Intensity가 높은 순으로 정렬 후 자르기
            if num_peaks > self.MAX_LEN:
                processed_peaks.sort(key=lambda p: p[1], reverse=True) # p[1] = norm_int
                num_peaks = self.MAX_LEN
                processed_peaks = processed_peaks[:num_peaks]
            
            peak_sequence[:num_peaks] = torch.tensor(processed_peaks, dtype=torch.float32)
            peak_mask[:num_peaks] = True
        
        return {
            'peak_sequence': peak_sequence,    # [700, 2] (log_mz, norm_int)
            'peak_mask': peak_mask,            # [700]
            'input_ids': tokenized_text['input_ids'].squeeze(0),
            'attention_mask': tokenized_text['attention_mask'].squeeze(0)
        }


def prepare_dataloaders():
    """
    (MODIFIED for "Path A")
    - 깨끗한 MassBank와 MoNA Parquet 로드
    """
    
    print("Loading datasets (CLEANED: Centroid-Only)...")
    try:
        df_massbank = pd.read_parquet(config.MASSBANK_FILE)
        df_mona = pd.read_parquet(config.MONA_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the MODIFIED create_parquet scripts with MAX_PEAK_COUNT filter.")
        return None, None
    except AttributeError:
        print("Error: config.py에 MASSBANK_FILE 또는 MONA_FILE 변수가 없습니다.")
        return None, None
        
    df_massbank = df_massbank.dropna(subset=['smiles'])
    df_mona = df_mona.dropna(subset=['smiles'])
    
    print("Performing Zero-Shot split on MassBank (for clean test set)...")
    unique_smiles_massbank = df_massbank['smiles'].unique()
    
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(unique_smiles_massbank)
    
    split_idx = int(len(unique_smiles_massbank) * config.TRAIN_TEST_SPLIT_RATIO)
    train_smiles_massbank = set(unique_smiles_massbank[:split_idx])
    test_smiles_massbank = set(unique_smiles_massbank[split_idx:])
    
    train_massbank_df = df_massbank[df_massbank['smiles'].isin(train_smiles_massbank)].reset_index(drop=True)
    test_massbank_df = df_massbank[df_massbank['smiles'].isin(test_smiles_massbank)].reset_index(drop=True)
    df_mona = df_mona.reset_index(drop=True) # MoNA는 모두 훈련용

    tokenizer = get_tokenizer()
    
    train_dataset_massbank = MassBankDataset(train_massbank_df, tokenizer, is_train=True)
    train_dataset_mona = MassBankDataset(df_mona, tokenizer, is_train=True)
    
    train_dataset = ConcatDataset([train_dataset_massbank, train_dataset_mona])
    
    test_dataset = MassBankDataset(test_massbank_df, tokenizer, is_train=False)
    
    len_massbank_train = len(train_dataset_massbank)
    len_mona_train = len(train_dataset_mona)
    
    print("-" * 80)
    print(f"Total Train Spectra (MassBank): {len_massbank_train:,} ({(len_massbank_train / len(train_dataset))*100:.1f}%)")
    print(f"Total Train Spectra (MoNA): {len_mona_train:,} ({(len_mona_train / len(train_dataset))*100:.1f}%)")
    print(f"Total Train Spectra (Combined): {len(train_dataset):,}")
    print(f"Total Test Spectra (MassBank ZSR): {len(test_dataset):,}")
    print("-" * 80)
    print("NOTE: Training *without* WeightedRandomSampler first.")
    print("-" * 80)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    # Test the dataloader
    print("Testing MassBank Dataloader (Path A - Peak Sequence)...")
    train_loader, test_loader = prepare_dataloaders()
    
    if train_loader:
        print("\n" + "="*80)
        print("Testing TRAIN batch loading...")
        train_batch = next(iter(train_loader))
        print(f"  Peak Sequence shape: {train_batch['peak_sequence'].shape}")
        print(f"  Peak Mask shape: {train_batch['peak_mask'].shape}")
        print(f"  Input IDs shape: {train_batch['input_ids'].shape}")
        
        print("\nTesting one text prompt (dynamically built):")
        tokenizer = get_tokenizer()
        print(tokenizer.decode(train_batch['input_ids'][0], skip_special_tokens=True))
        
        print("\nTesting one peak sequence (first 5 peaks):")
        print(train_batch['peak_sequence'][0, :5, :])
        print("\nTesting corresponding peak mask:")
        print(train_batch['peak_mask'][0, :5])

        print("\n" + "="*80)
        print("✓ Dataloader test passed!")