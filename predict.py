"""
Advanced Capability Intelligence Engine
Uses ensemble logic to provide highly responsive, granular percentages.
"""
import numpy as np, pandas as pd, joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from modules.preprocess import load_data, FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'intelligence_model.pkl')

def train_and_save():
    df = load_data()
    X = df[FEATURE_COLS].fillna(0)
    y = (df['5G_Capability'] == 'Yes').astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest for granular probability scores
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(Xtr, ytr)
    
    bundle = {'model': model}
    joblib.dump(bundle, MODEL_PATH)
    return bundle

def load_bundle():
    if os.path.exists(MODEL_PATH): 
        try:
            return joblib.load(MODEL_PATH)
        except:
            pass
    return train_and_save()

def predict_5g(units_sold, revenue, market_share, avg_speed, preference, subscribers, coverage):
    # Always ensure we get the dictionary bundle
    bundle = load_bundle()
    model = bundle['model']
    
    raw = np.array([[units_sold, revenue, market_share, avg_speed, preference, subscribers, coverage]])
    proba = model.predict_proba(raw)[0]
    pred = int(model.predict(raw)[0])
    label = '5G Capable' if pred == 1 else 'Standard Connectivity'
    return label, proba, pred
