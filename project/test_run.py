import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'project/src'))

from preprocess import generate_dataset, feature_engineering, normalize_features
from behaviour_score import calculate_behaviour_score
from model import InternetBehaviourModel
from evaluate import evaluate_model, rule_based_detection, generate_pdf_report

def run_verification():
    print("1. Generating Dataset...")
    df = generate_dataset(100)
    print(f"   Generated {len(df)} records.")
    
    print("2. Calculating Scores and Features...")
    df = calculate_behaviour_score(df)
    df = feature_engineering(df)
    
    print("3. Training Model...")
    iso_model = InternetBehaviourModel()
    feature_cols = iso_model.feature_cols
    df_norm = normalize_features(df, feature_cols)
    iso_model.train(df_norm)
    
    print("4. Predicting...")
    preds = iso_model.predict(df_norm)
    df['prediction'] = preds
    
    print("5. Evaluating...")
    metrics = evaluate_model(df['label'], df['prediction'])
    print("   Metrics:", metrics)
    
    print("6. Rule Based Comparison...")
    rule_preds = rule_based_detection(df)
    
    print("7. Generating PDF...")
    # Mocking rule metrics for PDF generation call
    rule_metrics = evaluate_model(df['label'], [1 if x=='normal' else -1 for x in rule_preds])
    generate_pdf_report(metrics, rule_metrics, metrics['Confusion Matrix'])
    
    if os.path.exists("project/evaluation_report.pdf"):
        print("   PDF Report successfully created.")
    else:
        print("   ERROR: PDF Report not found.")

if __name__ == "__main__":
    run_verification()
