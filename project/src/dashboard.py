import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_distribution(df, feature):
    """Returns a matplotlib figure for feature distribution."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=feature, hue='prediction', kde=True, ax=ax, palette={1: 'green', -1: 'red'})
    ax.set_title(f'Distribution of {feature}')
    return fig

def plot_anomaly_pie(df):
    """Returns a pie chart figure."""
    counts = df['prediction'].value_counts()
    # -1 is anomaly, 1 is normal
    labels = ['Normal' if i==1 else 'Anomaly' for i in counts.index]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    ax.set_title('Anomaly Ratio')
    return fig

def plot_user_activity(df):
    """Returns a line graph of activity (e.g. uploads over index)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    # Just plotting uploads for first 50 users to avoid clutter or all users
    subset = df.head(50)
    ax.plot(subset.index, subset['file_upload_count'], marker='o', label='Uploads')
    
    # Highlight anomalies
    anomalies = subset[subset['prediction'] == -1]
    ax.scatter(anomalies.index, anomalies['file_upload_count'], color='red', label='Anomaly', zorder=5)
    
    ax.set_title('User Activity (Upload Counts)')
    ax.set_xlabel('User Index')
    ax.set_ylabel('Upload Count')
    ax.legend()
    return fig
