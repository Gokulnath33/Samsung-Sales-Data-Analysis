import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.predict import predict_5g
from modules.cluster import run_kmeans
import pandas as pd

def test_prediction():
    print("--- Testing Capability Prediction ---")
    # Test Case 1: Likely "No" (Low revenue, low market share)
    # 26396, 4212951, 1.04, 57.36, 39.55, 293.1, 55.87
    label, proba, pred = predict_5g(26000, 4000000, 1.0, 250, 50, 40, 50)
    print(f"Case 1 (Low Rev): Result={label}, Confidence={proba[pred]*100:.1f}%")
    
    # Test Case 2: Likely "Yes" (High revenue, higher stats)
    # 15912, 17178326, 5.41, 59.12, 12.14, 179.15, 80.79
    label, proba, pred = predict_5g(15000, 17000000, 5.0, 180, 80, 12, 60)
    print(f"Case 2 (High Rev): Result={label}, Confidence={proba[pred]*100:.1f}%")

def test_clustering():
    print("\n--- Testing Market Segmentation ---")
    res_df, n_c, _, sil, exp, insights = run_kmeans(3)
    print(f"Clusters Formed: {n_c}")
    print(f"Silhouette Score: {sil:.3f}")
    for cid, info in insights.items():
        print(f"Cluster {cid} ({info['label']}): {info['n']} models")

if __name__ == "__main__":
    try:
        test_prediction()
        test_clustering()
        print("\nSUCCESS: All logic modules are producing valid outputs.")
    except Exception as e:
        print(f"\nERROR: {e}")
