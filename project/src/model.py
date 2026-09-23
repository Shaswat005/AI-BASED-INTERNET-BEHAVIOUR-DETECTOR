from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import os

class InternetBehaviourModel:
    def __init__(self, contamination='auto', n_estimators=200):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = [
            'login_time', 'logout_time', 'session_duration',
            'file_upload_count', 'file_download_count', 
            'ip_change_count', 'failed_login_attempts', 
            'work_hours_consistency', 'behaviour_score',
            'odd_hour_login', 'sudden_upload_spike', 'abnormal_ip_activity'
        ]

    def train(self, df):
        """Trains the Isolation Forest model."""
        X = df[self.feature_cols]
        self.model.fit(X)
        return self.model

    def predict(self, df):
        """Returns predictions: -1 (anomaly), 1 (normal)."""
        X = df[self.feature_cols]
        return self.model.predict(X)
        
    def save_model(self, path="models/iso_forest.pkl"):
        """Saves the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load_model(self, path="models/iso_forest.pkl"):
        """Loads a trained model."""
        self.model = joblib.load(path)
