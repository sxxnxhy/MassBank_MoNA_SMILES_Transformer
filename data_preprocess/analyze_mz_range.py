import pandas as pd
import numpy as np
from tqdm import tqdm

def analyze_mz_range(parquet_file="mona_30peaks_1064635.parquet"):
    """
    Analyzes the m/z range from the preprocessed parquet file.
    (MODIFIED: Fixed ValueError: ... ambiguous ...)
    """
    
    print(f"Loading {parquet_file}...")
    try:
        df = pd.read_parquet(parquet_file)
    except FileNotFoundError:
        print(f"Error: '{parquet_file}' not found.")
        print("Please run dataset_parser.py first.")
        return
    except ImportError:
        print("Error: 'pyarrow' or 'fastparquet' library not found.")
        print("Please install it: pip install pandas pyarrow")
        return

    print(f"Found {len(df)} total records.")
    
    min_mz_list = []
    max_mz_list = []

    print("Analyzing m/z ranges... (This may take a minute)")
    for peak_list in tqdm(df['peaks'], desc="Scanning peaks"):
        
        if peak_list is None or peak_list.size == 0:
            continue

        # peak_list is likely a numpy array of (mz, int) tuples/lists
        try:
            # Extract m/z values (the first element of each pair)
            mzs = [p[0] for p in peak_list]
            if mzs:
                min_mz_list.append(min(mzs))
                max_mz_list.append(max(mzs))
        except (IndexError, TypeError, ValueError) as e:
            # Handle potential malformed data inside the array
            print(f"Warning: Found malformed peak data: {e}...")
            continue
            
    if not min_mz_list:
        print("Error: No valid peak data found.")
        return

    # Numpy-fy for percentile calculation
    min_mz_arr = np.array(min_mz_list)
    max_mz_arr = np.array(max_mz_list)

    # --- Final Statistics ---
    print("\n" + "="*80)
    print("m/z Range Analysis Results (Percentiles):")
    print("="*80)
    
    print("\n--- Minimum m/z (Spectrum Start) ---")
    print(f"  0th percentile (Abs. Min): {np.percentile(min_mz_arr, 0):.2f}")
    print(f"  5th percentile (Recommended MIN): {np.percentile(min_mz_arr, 5):.2f}")
    print(f" 10th percentile:                 {np.percentile(min_mz_arr, 10):.2f}")
    print(f" 50th percentile (Median):        {np.percentile(min_mz_arr, 50):.2f}")

    print("\n--- Maximum m/z (Spectrum End) ---")
    print(f" 50th percentile (Median):        {np.percentile(max_mz_arr, 50):.2f}")
    print(f" 90th percentile:                 {np.percentile(max_mz_arr, 90):.2f}")
    print(f" 95th percentile (Recommended MAX): {np.percentile(max_mz_arr, 95):.2f}")
    print(f"100th percentile (Abs. Max):     {np.percentile(max_mz_arr, 100):.2f}")
    print("="*80)
    
    print("\nRecommendations:")
    print(f"  Set config.py 'MS_X_MIN' to: {np.percentile(min_mz_arr, 5):.0f}")
    print(f"  Set config.py 'MS_X_MAX' to: {np.percentile(max_mz_arr, 95):.0f}")
    print("="*80)

if __name__ == "__main__":
    analyze_mz_range()