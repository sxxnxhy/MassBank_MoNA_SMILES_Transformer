# create_mona_parquet.py
import ijson
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import sys
import decimal
import pyarrow
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 설정 ---
MONA_JSON_FILE = r"C:\Users\syoo\Downloads\MoNA-export-All_Spectra-json\MoNA-export-All_Spectra.json"
OUTPUT_PARQUET_FILE = "mona_preprocessed.parquet"
MINIMUM_QUALITY_SCORE = 3.0 
SMILES_PRIORITY = 'computed' 

# --- [NEW] 피크 개수 상한선 설정 ---
# 99백분위수(655)와 Max(361k)를 고려, 1000을 "프로파일" 컷오프로 설정
MAX_PEAK_COUNT = 10
# --- [END NEW] ---

def parse_spectrum_string(spectrum_str):
    """ (수정 없음) Python 리스트를 반환 """
    peaks_list = [] 
    if not spectrum_str or not isinstance(spectrum_str, str):
        return None
    try:
        for peak in spectrum_str.strip().split(' '):
            mz, intensity = peak.split(':')
            peaks_list.append((float(mz), float(intensity)))
        if not peaks_list:
            return None
        return peaks_list 
    except Exception:
        return None

def extract_smiles(compound_metadata, priority='computed'):
    """ (수정 없음) SMILES 추출 """
    smiles_computed = None
    smiles_none = None
    if not isinstance(compound_metadata, list):
        return None
    for item in compound_metadata:
        if item.get('name') == 'SMILES':
            if item.get('category') == 'computed':
                smiles_computed = item.get('value')
            elif item.get('category') == 'none':
                smiles_none = item.get('value')
    if priority == 'computed':
        return smiles_computed or smiles_none
    else:
        return smiles_none or smiles_computed

def main():
    logging.info(f"Starting MoNA JSON parsing (Streaming)...")
    
    processed_data = []
    total_spectra = 0
    filtered_quality = 0
    filtered_rdkit = 0
    filtered_missing = 0
    filtered_malformed = 0
    filtered_peak_count = 0 # <-- [NEW]

    try:
        with open(MONA_JSON_FILE, 'rb') as f:
            parser = ijson.items(f, 'item')
            
            for spectrum_obj in tqdm(parser, desc="Parsing MoNA (ijson)"):
                total_spectra += 1
                try:
                    # 1. 품질 필터링
                    score_obj = spectrum_obj.get('score', {})
                    if score_obj is None:
                        filtered_quality += 1
                        continue
                    score = float(score_obj.get('score', 0))
                    if score < MINIMUM_QUALITY_SCORE:
                        filtered_quality += 1
                        continue

                    # 2. SMILES 추출 및 검사
                    compound_list = spectrum_obj.get('compound')
                    if not compound_list or not isinstance(compound_list, list) or len(compound_list) == 0:
                        filtered_missing += 1
                        continue
                    compound_metadata = compound_list[0].get('metaData')
                    if not compound_metadata:
                        filtered_missing += 1
                        continue
                    smiles = extract_smiles(compound_metadata, priority=SMILES_PRIORITY)
                    if not smiles or not isinstance(smiles, str) or smiles.strip() == "":
                        filtered_missing += 1
                        continue
                    if Chem.MolFromSmiles(smiles) is None:
                        filtered_rdkit += 1
                        continue

                    # 3. 피크 문자열 파싱
                    peaks_str = spectrum_obj.get('spectrum')
                    peaks_list = parse_spectrum_string(peaks_str)
                    
                    if peaks_list is None:
                        filtered_missing += 1
                        continue

                    # --- [NEW] 피크 개수 필터링 (프로파일 제거) ---
                    if len(peaks_list) > MAX_PEAK_COUNT:
                        filtered_peak_count += 1
                        continue # 피크 개수가 1000개를 초과하면 삭제
                    # --- [END NEW] ---

                    # 5. 저장
                    processed_data.append({
                        'smiles': smiles,
                        'peaks': peaks_list
                    })
                
                except (KeyError, TypeError, AttributeError, decimal.InvalidOperation) as e:
                    filtered_malformed += 1
                    continue

    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {MONA_JSON_FILE}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\nAn error occurred during parsing at spectrum #{total_spectra}: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("Parsing Complete.")
    print(f"Total spectra processed: {total_spectra:,}")
    print(f"  Kept spectra: {len(processed_data):,}")
    print(f"  Filtered (Missing/Quality/RDKit): {(filtered_missing + filtered_quality + filtered_rdkit):,}")
    print(f"  Filtered (Malformed JSON): {filtered_malformed:,}")
    print(f"  Filtered (Peak Count > {MAX_PEAK_COUNT}): {filtered_peak_count:,}") # <-- [NEW]
    print("="*80)
    
    if not processed_data:
        logging.error("Error: No data was processed.")
        sys.exit(1)

    logging.info("Converting to Pandas DataFrame...")
    df = pd.DataFrame(processed_data)

    logging.info(f"Saving to {OUTPUT_PARQUET_FILE}...")
    try:
        # (원본과 동일) pyarrow가 'list'를 잘 저장함
        df.to_parquet(OUTPUT_PARQUET_FILE, index=False)
    except Exception as e:
        logging.error(f"Failed to save parquet file: {e}")
        sys.exit(1)

    logging.info("✅ Done.")

if __name__ == "__main__":
    main()