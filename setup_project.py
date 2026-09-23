import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'project/src'))

from preprocess import generate_dataset, feature_engineering, normalize_features
from behaviour_score import calculate_behaviour_score
from model import InternetBehaviourModel

def setup_assets():
    print("Initializing Project Assets...")
    
    # 1. Generate and Save Data
    print("Generating synthetic dataset...")
    df = generate_dataset(500)
    df = calculate_behaviour_score(df)
    df = feature_engineering(df)
    
    data_path = "project/data/dataset.csv"
    df.to_csv(data_path, index=False)
    print(f"Saved dataset to {data_path}")
    
    # 2. Train and Save Model
    print("Training Isolation Forest model...")
    iso_model = InternetBehaviourModel()
    
    feature_cols = iso_model.feature_cols
    df_norm = normalize_features(df, feature_cols)
    
    iso_model.train(df_norm)
    
    model_path = "project/models/iso_forest.pkl"
    iso_model.save_model(model_path)
    print(f"Saved model to {model_path}")

    print("Setup Complete.")

if __name__ == "__main__":
    setup_assets()
