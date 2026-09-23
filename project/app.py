import streamlit as st
import pandas as pd
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import generate_dataset, feature_engineering, normalize_features
from behaviour_score import calculate_behaviour_score
from model import InternetBehaviourModel
from evaluate import evaluate_model, rule_based_detection, generate_pdf_report
import dashboard

# 7. Logging System
logging.basicConfig(filename='predictions.log', level=logging.INFO, 
                    format='%(message)s')

def log_prediction(row, pred):
    """Logs prediction to file."""
    # pred is 1 (normal) or -1 (anomaly)
    pred_str = "Normal" if pred == 1 else "Suspicious"
    log_msg = f"{datetime.now()},{row['user_id']},{row['behaviour_score']},{pred_str}"
    logging.info(log_msg)

def main():
    st.set_page_config(page_title="AI Threat Detector", layout="wide")
    st.title("AI-Based Internet Behaviour Threat Detector")
    
    st.sidebar.header("Options")
    data_source = st.sidebar.radio("Data Source", ["Generate Synthetic", "Upload CSV"])
    
    df = None
    
    if data_source == "Generate Synthetic":
        if st.sidebar.button("Generate & Process"):
            with st.spinner("Generating data..."):
                raw_df = generate_dataset(500)
                df = calculate_behaviour_score(raw_df)
                df = feature_engineering(df)
                st.session_state['df'] = df
            st.success("Data Generated!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            # Ensure required columns exist
            # For simplicity, we assume uploaded CSV has raw structure.
            df = calculate_behaviour_score(raw_df)
            df = feature_engineering(df)
            st.session_state['df'] = df
            
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Train / Predict
        st.subheader("Model Training & Prediction")
        
        iso_model = InternetBehaviourModel()
        
        # Normalize for model
        feature_cols = iso_model.feature_cols
        df_norm = normalize_features(df, feature_cols)
        
        iso_model.train(df_norm)
        preds = iso_model.predict(df_norm)
        
        df['prediction'] = preds
        df['prediction_label'] = df['prediction'].apply(lambda x: 'Normal' if x==1 else 'Suspicious')
        
        # Logging
        # Only log new predictions in a real app, here we log all for demo
        # preventing massive file write on every rerun:
        if 'logged' not in st.session_state:
            for idx, row in df.iterrows():
                log_prediction(row, row['prediction'])
            st.session_state['logged'] = True
            
        # Metrics
        st.subheader("Evaluation Metrics")
        col1, col2 = st.columns(2)
        
        if 'label' in df.columns:
            metrics_iso = evaluate_model(df['label'], df['prediction'])
            
            # Rule Based Comparison
            rule_preds = rule_based_detection(df)
            metrics_rule = evaluate_model(df['label'], [1 if x=='normal' else -1 for x in rule_preds])
            
            with col1:
                st.markdown("### Isolation Forest Results")
                st.json({k:v for k,v in metrics_iso.items() if k!='Confusion Matrix'})
                
            with col2:
                st.markdown("### Rule-Based Results")
                st.json({k:v for k,v in metrics_rule.items() if k!='Confusion Matrix'})
                
            # PDF Report
            if st.button("Generate PDF Report"):
                generate_pdf_report(metrics_iso, metrics_rule, metrics_iso['Confusion Matrix'])
                st.success("Report saved to project/evaluation_report.pdf")
                with open("project/evaluation_report.pdf", "rb") as pdf_file:
                    st.download_button("Download Report", pdf_file, "report.pdf")

        # Visualizations
        st.subheader("Dashboard")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(dashboard.plot_anomaly_pie(df))
        with c2:
            st.pyplot(dashboard.plot_user_activity(df))
            
        st.pyplot(dashboard.plot_feature_distribution(df, 'behaviour_score'))
        
        # Data View
        st.subheader("Prediction Data")
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Export Predictions to CSV", csv, "predictions.csv", "text/csv")

if __name__ == "__main__":
    main()
