import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import generate_dataset

def main():
    print("Generating dataset with 1000 records...")
    df = generate_dataset(1000)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path} with {len(df)} records.")

if __name__ == "__main__":
    main()
