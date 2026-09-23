<<<<<<< HEAD
# AI-Based Internet Behaviour Threat Detector

## Overview
This project uses the **Isolation Forest** algorithm to detect anomalous user behavior in an internet usage dataset. It compares the machine learning approach with a traditional rule-based system.

## Features
- **Dataset Generation**: Creates synthetic data with normal and anomalous patterns.
- **Feature Engineering**: Calculates metrics like `odd_hour_login`, `sudden_upload_spike`, and a custom `behaviour_score`.
- **Model**: Isolation Forest for unsupervised anomaly detection.
- **Evaluation**: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
- **Dashboard**: Interactive Streamlit app for visualization and reporting.
- **Reporting**: PDF export of results and logging of predictions.

## Folder Structure
```
project/
├── data/
├── models/
├── src/
│   ├── preprocess.py
│   ├── behaviour_score.py
│   ├── model.py
│   ├── evaluate.py
│   └── dashboard.py
├── app.py
├── requirements.txt
└── predictions.log
```

## Setup & Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Project** (Populates `data/` and `models/`):
   ```bash
   python setup_project.py
   ```

3. **Run Dashboard**:
   ```bash
   streamlit run app.py
   ```

3. **Check Results**:
   - The app will generate `predictions.log`.
   - You can download a PDF report from the dashboard.

## Methodology
The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. Anomalies are susceptible to isolation and will have shorter path lengths in the trees.

**Behaviour Score**:
- Odd-hour login: +20
- High file uploads: +30
- High IP change: +25
- Failed logins: +15
Normalized to 0-100.

## Accuracy Verification
We compare the labels generated during synthesis (ground truth) against the model's predictions. The rule-based system serves as a baseline.
=======
# AI-BASED-INTERNET-BEHAVIOUR-DETECTOR
>>>>>>> 353feb5dc53b5a2c7460745e689f5ba24d0f18ca
