import pandas as pd
import numpy as np
from datetime import datetime, time
import random

def generate_dataset(num_records=500):
    """Generates a synthetic user-behaviour dataset."""
    data = []
    
    for i in range(num_records):
        # 1. Login Time (0-24 hours in decimal)
        # Normal shifts: 8 AM - 6 PM, Anomalous: < 5 AM or > 10 PM
        if random.random() < 0.9: # 90% normal
            login_hour = random.uniform(8, 18)
        else:
            login_hour = random.choice([random.uniform(0, 5), random.uniform(22, 24)])
            
        # 2. Logout Time & Session Duration
        # Duration: Normal 1-9 hours, Anomalous > 12 hours or very short < 10 mins
        if random.random() < 0.9:
            duration = random.uniform(1, 9)
        else:
            duration = random.choice([random.uniform(0.1, 0.5), random.uniform(12, 16)])
        
        logout_hour = (login_hour + duration) % 24
        
        # 3. File Uploads/Downloads
        if random.random() < 0.9:
            uploads = int(np.random.exponential(2))  # Mostly low
            downloads = int(np.random.normal(10, 5))
        else:
            uploads = int(random.uniform(10, 50)) # Spike
            downloads = int(random.uniform(50, 200))

        # 4. IP Change Count
        if random.random() < 0.95:
            ip_changes = random.randint(0, 1)
        else:
            ip_changes = random.randint(3, 10)
            
        # 5. Failed Login Attempts
        if random.random() < 0.9:
            failed_logins = 0
        else:
            failed_logins = random.randint(3, 10)
            
        # 6. Work Hours Consistency (0-1)
        # Higher is better. Low consistency might indicate shared account or erratic behavior.
        consistency = random.uniform(0.5, 1.0) if random.random() < 0.9 else random.uniform(0.0, 0.4)
        
        # Ground Truth Label Integration (for testing purposes)
        # We define "suspicious" based on the injected anomalies for validation
        is_suspicious = (
            (login_hour < 5 or login_hour > 22) or
            (uploads > 15) or
            (ip_changes > 2) or
            (failed_logins > 5)
        )
        label = "suspicious" if is_suspicious else "normal"

        data.append({
            "user_id": f"user_{random.randint(1, 50)}",
            "login_time": login_hour,
            "logout_time": logout_hour,
            "session_duration": duration,
            "file_upload_count": uploads,
            "file_download_count": downloads,
            "ip_change_count": ip_changes,
            "failed_login_attempts": failed_logins,
            "work_hours_consistency": consistency,
            "label": label
        })
        
    df = pd.DataFrame(data)
    return df

def feature_engineering(df):
    """Performs feature engineering and normalization."""
    df_processed = df.copy()
    
    # 1. Odd Hour Login (1 if < 5 AM or > 10 PM)
    # 10 PM is 22.0
    df_processed['odd_hour_login'] = df_processed['login_time'].apply(
        lambda x: 1 if (x < 5 or x > 22) else 0
    )
    
    # 2. Sudden Upload Spike
    avg_upload = df_processed['file_upload_count'].mean()
    std_upload = df_processed['file_upload_count'].std()
    threshold = avg_upload + 2 * std_upload
    df_processed['sudden_upload_spike'] = df_processed['file_upload_count'].apply(
        lambda x: 1 if x > threshold else 0
    )
    
    # 3. Abnormal IP Activity
    # Simple heuristic: more than 2 IP changes is abnormal
    df_processed['abnormal_ip_activity'] = df_processed['ip_change_count'].apply(
        lambda x: 1 if x > 2 else 0
    )
    
    return df_processed

def normalize_features(df, feature_cols):
    """Normalizes numerical features using Min-Max scaling or Standardization."""
    # Using simple min-max for this implementation to keep 0-1 ranges consistent
    df_norm = df.copy()
    for col in feature_cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val - min_val > 0:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    return df_norm
