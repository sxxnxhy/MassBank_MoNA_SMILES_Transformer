import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
import config
from tqdm import tqdm
from rdkit import Chem

def get_tokenizer():
    """Initializes and returns the tokenizer."""
    return AutoTokenizer.from_pretrained(
            config.TEXT_ENCODER['model_name']
        )

class MassBankDataset(Dataset):
    """
    (MODIFIED)
    - Intensity: SQRT normalization (Handles Power Law)
    - m/z: Log transform (Handles Scale)
    """
    def __init__(self, df, tokenizer, is_train=False): 
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.MAX_LEN = config.MAX_PEAK_SEQ_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row.get('smiles')
        text_prompt = f"{smiles}"
        
        tokenized_text = self.tokenizer(
            text_prompt,
            max_length=config.TEXT_ENCODER['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        peak_list = row['peaks'] 
        if isinstance(peak_list, np.ndarray):
            peak_list = peak_list.tolist()

        processed_peaks = []
        
        # --- [FIX 2: Intensity Preprocessing (Power Law)] ---
        if peak_list:
            # 1. Extract intensities
            intensities = np.array([p[1] for p in peak_list])
            # 2. Apply Square Root (Boosts small diagnostic peaks)
            #    e.g., 1% peak (0.01) becomes 10% (0.1)
            root_intensities = np.sqrt(intensities)
            # 3. Max Scaling
            max_int = root_intensities.max() if len(root_intensities) > 0 else 1.0
            if max_int < 1e-9: max_int = 1.0
            norm_intensities = root_intensities / max_int
            
            # Extract m/z
            mzs = np.array([p[0] for p in peak_list])
            # Log transform m/z for numerical stability
            log_mzs = np.log(mzs + 1.0)
            
            # Combine
            for mz, inten in zip(log_mzs, norm_intensities):
                processed_peaks.append((mz, inten))
        
        # --- 3. Padding / Sorting ---
        num_peaks = len(processed_peaks)
        peak_sequence = torch.zeros(self.MAX_LEN, 2, dtype=torch.float32)
        peak_mask = torch.zeros(self.MAX_LEN, dtype=torch.bool)

        if num_peaks > 0:
            if num_peaks > self.MAX_LEN:
                # Sort by intensity (descending) and truncate
                processed_peaks.sort(key=lambda p: p[1], reverse=True)
                num_peaks = self.MAX_LEN
                processed_peaks = processed_peaks[:num_peaks]
            
            # Sort by m/z for the Transformer (Positional consistency)
            # Although Fourier Features handle random order, sorting helps the attention mechanism 
            # learn "local" spectral features (like isotopic clusters)
            processed_peaks.sort(key=lambda p: p[0])
            
            peak_sequence[:num_peaks] = torch.tensor(processed_peaks, dtype=torch.float32)
            peak_mask[:num_peaks] = True
        
        return {
            'peak_sequence': peak_sequence,
            'peak_mask': peak_mask,
            'input_ids': tokenized_text['input_ids'].squeeze(0),
            'attention_mask': tokenized_text['attention_mask'].squeeze(0)
        }


def prepare_dataloaders():
    """
    (MODIFIED)
    - Fixes Data Leakage by splitting on InChIKey (First Block)
    """
    print("Loading datasets...")
    try:
        df_massbank = pd.read_parquet(config.MASSBANK_FILE)
        df_mona = pd.read_parquet(config.MONA_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
        
    df_massbank = df_massbank.dropna(subset=['smiles'])
    df_mona = df_mona.dropna(subset=['smiles'])
    
    # --- [FIX 3: InChIKey Splitting to prevent Leakage] ---
    print("Generating InChIKeys for MassBank split (Preventing Stereoisomer Leakage)...")
    
    # Helper to get first block
    def get_inchikey_block1(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                key = Chem.MolToInchiKey(mol)
                return key.split('-')[0] # Connectivity layer only
        except:
            return None
        return None

    # Apply to unique SMILES only (faster)
    unique_smiles = pd.DataFrame(df_massbank['smiles'].unique(), columns=['smiles'])
    tqdm.pandas(desc="Calculating InChIKeys")
    unique_smiles['inchikey_1'] = unique_smiles['smiles'].progress_apply(get_inchikey_block1)
    
    # Remove failed conversions
    unique_smiles = unique_smiles.dropna()
    
    # Split based on Unique InChIKey Blocks (Connectivity)
    unique_blocks = unique_smiles['inchikey_1'].unique()
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(unique_blocks)
    
    split_idx = int(len(unique_blocks) * config.TRAIN_TEST_SPLIT_RATIO)
    train_blocks = set(unique_blocks[:split_idx])
    test_blocks = set(unique_blocks[split_idx:])
    
    # Map back to SMILES
    train_smiles = set(unique_smiles[unique_smiles['inchikey_1'].isin(train_blocks)]['smiles'])
    test_smiles = set(unique_smiles[unique_smiles['inchikey_1'].isin(test_blocks)]['smiles'])
    
    # Create DataFrames
    train_massbank_df = df_massbank[df_massbank['smiles'].isin(train_smiles)].reset_index(drop=True)
    test_massbank_df = df_massbank[df_massbank['smiles'].isin(test_smiles)].reset_index(drop=True)
    df_mona = df_mona.reset_index(drop=True)

    print(f"  MassBank Train (Connectivity): {len(train_blocks)} blocks -> {len(train_massbank_df)} spectra")
    print(f"  MassBank Test (Connectivity): {len(test_blocks)} blocks -> {len(test_massbank_df)} spectra")
    
    tokenizer = get_tokenizer()
    
    train_dataset_massbank = MassBankDataset(train_massbank_df, tokenizer, is_train=True)
    # train_dataset_mona = MassBankDataset(df_mona, tokenizer, is_train=True)
    

    train_dataset = ConcatDataset([train_dataset_massbank]) # Removed train_dataset_mona
    #train_dataset = ConcatDataset([train_dataset_massbank, train_dataset_mona])
    
    test_dataset = MassBankDataset(test_massbank_df, tokenizer, is_train=False)
    
    len_massbank_train = len(train_dataset_massbank)
    # len_mona_train = len(train_dataset_mona)
    
    print("-" * 80)
    print(f"Total Train Spectra (MassBank): {len_massbank_train:,} ({(len_massbank_train / len(train_dataset))*100:.1f}%)")
    # print(f"Total Train Spectra (MoNA): {len_mona_train:,} ({(len_mona_train / len(train_dataset))*100:.1f}%)")
    print(f"Total Train Spectra (Combined): {len(train_dataset):,}")
    print(f"Total Test Spectra (MassBank ZSR): {len(test_dataset):,}")
    print("-" * 80)
    # print("NOTE: Training *without* WeightedRandomSampler first.")
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