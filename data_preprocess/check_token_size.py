import pandas as pd
import numpy as np
from transformers import AutoTokenizer

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from tqdm import tqdm

# --- 1. 토큰화 함수 (재현) ---
def tokenize_and_check(text, tokenizer, max_len):
    """텍스트를 토큰화하고, 길이가 max_len을 초과하는지 확인합니다."""
    # `truncation=False`로 설정하여 원본 길이를 확인합니다.
    encoding = tokenizer(
        text,
        truncation=False,
        padding=False,
        return_tensors='pt'
    )
    
    # [CLS]와 [SEP] 토큰 (2개)를 제외한 실제 토큰 수
    token_count = encoding.input_ids.shape[1]
    
    return token_count, token_count > max_len

# --- 2. 메인 분석 함수 ---
def run_truncation_check():
    
    max_length = 369 # 256
    tokenizer = AutoTokenizer.from_pretrained(
            config.TEXT_ENCODER['model_name'],
            do_lower_case=False
        )

    # 데이터 로딩 (전체 데이터셋 필요)
    # df = pd.read_parquet('mona_30peaks_1064635.parquet')
    df = pd.read_parquet('massbank_300peaks.parquet')
    print(f"Checking {len(df)} records for token length...")

    # 동적 프롬프트 생성 및 길이 확인
    truncated_count = 0
    max_tokens_found = 0
    
    for idx in tqdm(range(len(df)), desc="Checking Prompts"):
        row = df.iloc[idx]
        
        # --- [dataset.py의 get_item 로직 복사] ---
        # 이 부분이 프롬프트를 생성하는 로직입니다.
        def get_safe_string(row, key):
            val = row.get(key)
            if pd.isna(val) or val is None or str(val).strip() == "":
                return None
            return str(val).strip()

        smiles = get_safe_string(row, 'smiles')
        iupac = get_safe_string(row, 'iupac')

        prompt_parts = []
        
        prompt_parts.append(f"{smiles}")

        text_prompt = " ".join(prompt_parts)
   
        
        # 3. 토큰화 및 비교
        token_count, is_truncated = tokenize_and_check(text_prompt, tokenizer, max_length)
        
        if is_truncated:
            truncated_count += 1
        
        if token_count > max_tokens_found:
            max_tokens_found = token_count

    # --- 4. 결과 출력 ---
    print("\n" + "="*80)
    print("Truncation Analysis:")
    print(f"Target Max Length (L): {max_length}")
    print(f"Max Tokens Found:      {max_tokens_found}")
    print(f"Total Records:         {len(df):,}")
    print(f"Truncated Records:     {truncated_count:,}")
    
    if len(df) > 0:
        percent_truncated = (truncated_count / len(df)) * 100
        print(f"Percentage Truncated:  {percent_truncated:.3f}%")

    if truncated_count > 0:
        print("\n⚠️ WARNING: Truncation occurred!")
        print(f"To prevent data loss, increase MAX_LENGTH to at least {max_tokens_found + 5} in config.py.")
    else:
        print("\n✅ SUCCESS: No truncation detected!")
    print("="*80)

if __name__ == '__main__':
    run_truncation_check()