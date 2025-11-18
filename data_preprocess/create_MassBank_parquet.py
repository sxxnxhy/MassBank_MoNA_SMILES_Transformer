# create_parquet.py
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import pyarrow

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- [수정] 피크 개수 상/하한선 설정 (1 ~ 30개) ---
MIN_PEAK_COUNT = 1   
MAX_PEAK_COUNT = 30 
# --- [END 수정] ---

def parse_record(file_path, domain_name):
    smiles = None
    iupac = None
    peak_data = [] # Python 리스트
    instrument_type = None
    instrument = None
    is_reading_peaks = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith("//"):
                    if is_reading_peaks:
                        is_reading_peaks = False
                    continue
                
                if is_reading_peaks:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            peak_data.append((mz, intensity))
                    except (ValueError, IndexError):
                        continue
                    continue

                # (SMILES, IUPAC, INSTRUMENT 등 파싱...)
                if line.startswith("CH$SMILES:"):
                    smiles = line.split(":", 1)[1].strip()
                elif line.startswith("CH$IUPAC:"):
                    iupac = line.split(":", 1)[1].strip()
                elif line.startswith("AC$INSTRUMENT_TYPE:"):
                    instrument_type = line.split(":", 1)[1].strip()
                elif line.startswith("AC$INSTRUMENT:"):
                    instrument = line.split(":", 1)[1].strip()
                elif line.startswith("PK$PEAK:"):
                    is_reading_peaks = True

    except Exception as e:
        logging.warning(f"Error reading {file_path}: {e}")
        return None

    # --- [FILTER 1] 필수 정보 누락 (피크 0개 포함) ---
    if not smiles or not peak_data:
        return None

    # --- [FILTER 2] 피크 개수 필터링 (1 ~ 30개) ---
    peak_count = len(peak_data)
    if peak_count > MAX_PEAK_COUNT or peak_count < MIN_PEAK_COUNT:
        return None # 피크 개수가 1~30개 범위를 벗어남
    # --- [END FILTER 2] ---
        
    return {
        "smiles": smiles,
        "iupac": iupac,
        "peaks": peak_data,
        "domain_folder": domain_name,
        "instrument_type": instrument_type,
        "instrument": instrument
    }

def main():
    root_dir = Path("./data") # 경로 확인 필요
    if not root_dir.exists():
        logging.error(f"Error: Directory not found: {root_dir}")
        return

    logging.info(f"Starting scan of {root_dir}...")
    all_records = []
    total_files = 0
    filtered_count = 0

    for dirpath, _, filenames in tqdm(os.walk(root_dir), desc="Scanning folders"):
        domain_name = Path(dirpath).name
        for filename in filenames:
            if filename.endswith(".txt"):
                total_files += 1
                file_path = Path(dirpath) / filename
                record = parse_record(file_path, domain_name)
                
                if record:
                    all_records.append(record)
                else:
                    filtered_count += 1

    # logging.info(f"\nScan complete. Total files: {total_files:,}. Kept records: {len(all_records)::,}. Filtered (Range {MIN_PEAK_COUNT}-{MAX_PEAK_COUNT} / Missing): {filtered_count:,}")
    
    if not all_records:
        logging.warning("No valid records found.")
        return

    df = pd.DataFrame(all_records)
    output_file = "massbank_preprocessed.parquet" # 저장 경로 확인 필요
    try:
        df.to_parquet(output_file, index=False) 
        logging.info(f"\n✅ Preprocessing complete! Data saved to: {output_file}")
    except Exception as e:
        logging.error(f"Failed to save parquet file: {e}")
        return

    # (데이터 요약 부분은 동일)
    print("\n" + "="*80)
    print("Data Summary (Filtered):")
    print(f"Total Records (SMILES+Peaks): {len(df)}")
    # ... (이하 생략) ...

if __name__ == "__main__":
    main()