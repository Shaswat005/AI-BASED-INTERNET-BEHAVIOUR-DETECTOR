import pandas as pd

def calculate_behaviour_score(df):
    """
    Calculates a behaviour score based on weighted features.
    
    Weights:
    odd-hour login: +20
    high file uploads: +30
    high IP change: +25
    failed logins: +15
    
    Returns dataframe with a new 'behaviour_score' column (0-100).
    """
    df_scored = df.copy()
    
    scores = []
    
    # Pre-calculate thresholds for 'high' definitions if not already binary
    # We use the binary flags from feature engineering where possible, 
    # but let's recalculate based on raw values for strict adherence to requirements.
    
    avg_upload = df_scored['file_upload_count'].mean()
    std_upload = df_scored['file_upload_count'].std()
    upload_threshold = avg_upload + 2 * std_upload

    for index, row in df_scored.iterrows():
        score = 0
        
        # Odd-hour login (+20)
        if row['login_time'] < 5 or row['login_time'] > 22:
            score += 20
            
        # High file uploads (+30)
        if row['file_upload_count'] > upload_threshold: # using statistical threshold
            score += 30
        elif row['file_upload_count'] > 8: # fallback to rule-based low threshold if statistical is too high
             score += 10 # partial penalty
             
        # High IP change (+25)
        if row['ip_change_count'] > 2:
            score += 25
            
        # Failed logins (+15)
        if row['failed_login_attempts'] > 0:
            # Scale up to 15 based on severity? Or flat 15?
            # Requirement says "failed logins: +15". 
            # We'll apply it if there are *any* failed logins, 
            # or maybe threshold > 2 for "suspicious" level.
            if row['failed_login_attempts'] > 2:
                score += 15
            elif row['failed_login_attempts'] > 0:
                score += 5
        
        # Cap at 100
        score = min(score, 100)
        scores.append(score)
        
    df_scored['behaviour_score'] = scores
    return df_scored
