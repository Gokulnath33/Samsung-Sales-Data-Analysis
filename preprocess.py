import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'our dataset', 'samsung.csv')

FEATURE_COLS = ['Units_Sold', 'Revenue($)', 'Market_Share(%)', 'Avg_5GSpeed(Mbps)',
                'Preference_for_5G(%)', '5G_Subscribers(millions)', 'Regional _5GCoverage(%)']

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ensure relevant columns are numeric
    for col in FEATURE_COLS:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    return df

def get_features_target(df):
    X = df[FEATURE_COLS].copy().fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df['5G_Capability'])
    return X, y, le

def scale_features(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler
