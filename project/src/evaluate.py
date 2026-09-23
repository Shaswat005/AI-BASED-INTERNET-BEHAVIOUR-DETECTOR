from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from fpdf import FPDF

def evaluate_model(y_true, y_pred):
    """
    Calculates accuracy, precision, recall, f1, and confusion matrix.
    y_true: 'normal' or 'suspicious'
    y_pred: 1 (normal) or -1 (anomaly/suspicious)
    """
    # Convert y_pred to labels matches y_true
    y_pred_labels = ['normal' if x == 1 else 'suspicious' for x in y_pred]
    
    # We treat 'suspicious' as the Positive class for these metrics usually, 
    # but let's standardize on:
    # y_true binary: normal=0, suspicious=1
    # y_pred binary: 1=>0, -1=>1
    
    y_true_bin = [1 if x == 'suspicious' else 0 for x in y_true]
    y_pred_bin = [1 if x == 'suspicious' else 0 for x in y_pred_labels]
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Confusion Matrix": cm
    }

def rule_based_detection(df):
    """
    Rule: mark user suspicious if uploads > 8 OR ip_change_count > 2 OR login_time < 5 AM
    Returns list of labels ('normal' or 'suspicious')
    """
    predictions = []
    for _, row in df.iterrows():
        if (row['file_upload_count'] > 8) or \
           (row['ip_change_count'] > 2) or \
           (row['login_time'] < 5):
            predictions.append('suspicious')
        else:
            predictions.append('normal')
    return predictions

def generate_pdf_report(metrics_iso, metrics_rule, cm_iso):
    """Generates a PDF report with metrics and plots."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="AI-Based Internet Behaviour Threat Detector Report", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Isolation Forest Metrics:", ln=1)
    for k, v in metrics_iso.items():
        if k != "Confusion Matrix":
            pdf.cell(200, 10, txt=f"{k}: {v:.4f}", ln=1)
            
    pdf.ln(5)
    pdf.cell(200, 10, txt="Rule-Based Metrics:", ln=1)
    for k, v in metrics_rule.items():
        if k != "Confusion Matrix":
            pdf.cell(200, 10, txt=f"{k}: {v:.4f}", ln=1)
            
    # Save confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Suspicious'], yticklabels=['Normal', 'Suspicious'])
    plt.title('Isolation Forest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cm_plot.png')
    plt.close()
    
    pdf.ln(10)
    pdf.image('cm_plot.png', x=10, y=None, w=100)
    
    pdf.output("project/evaluation_report.pdf")
    # Clean up
    import os
    if os.path.exists('cm_plot.png'):
        os.remove('cm_plot.png')
