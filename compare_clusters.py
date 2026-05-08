import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from modules.preprocess import load_data, get_features_target, scale_features

def compare_clusters():
    df = load_data()
    X, _, _ = get_features_target(df)
    X_scaled, _ = scale_features(X)
    
    results = {}
    
    # 1. K-Means++ (K=3)
    km = KMeans(n_clusters=3, init='k-means++', random_state=42)
    km_labels = km.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)
    results['K-Means++'] = km_sil
    
    # 2. Agglomerative (K=3)
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)
    results['Agglomerative'] = agg_sil
    
    # 3. DBSCAN
    db = DBSCAN(eps=0.5, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    # Silhouette score only works if there's more than 1 cluster (excluding noise)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_clusters > 1:
        db_sil = silhouette_score(X_scaled, db_labels)
        results['DBSCAN'] = db_sil
    else:
        results['DBSCAN'] = -1 # Failed to find clusters
        
    return results

if __name__ == "__main__":
    print("--- Clustering Algorithm Comparison ---")
    scores = compare_clusters()
    for alg, score in scores.items():
        if score == -1:
            print(f"{alg}: Failed to find meaningful clusters with current density settings.")
        else:
            print(f"{alg}: Silhouette Score = {score:.4f}")
    
    best = max(scores, key=scores.get)
    print(f"\nRecommended Algorithm for Samsung Dataset: {best}")
