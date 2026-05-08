import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from modules.preprocess import load_data, get_features_target, scale_features

def run_segmentation(n_clusters=3):
    df = load_data()
    X, y, le = get_features_target(df)
    X_scaled, scaler = scale_features(X)
    
    # Exclusively K-Means++ as requested
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = model.fit_predict(X_scaled)
    
    # Metrics
    sil = silhouette_score(X_scaled, labels)
        
    # PCA for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    
    res_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    res_df['Cluster'] = labels.astype(str)
    res_df['5G_Capability'] = df['5G_Capability']
    
    # Generate Insights
    df['Cluster'] = labels
    insights = {}
    unique_clusters = sorted(set(labels))
    
    for c in unique_clusters:
        c_data = df[df['Cluster'] == c]
        avg_rev = c_data['Revenue($)'].mean()
        avg_units = c_data['Units_Sold'].mean()
        pct_5g = (c_data['5G_Capability'] == 'Yes').mean() * 100
        
        # Categorize
        if avg_rev > df['Revenue($)'].mean() and pct_5g > 70:
            lbl, note = "Premium 5G Leaders", "High revenue models with dominant 5G adoption."
        elif avg_rev < df['Revenue($)'].mean() and pct_5g < 30:
            lbl, note = "Standard Legacy Tier", "Budget-focused models with limited 5G connectivity."
        else:
            lbl, note = "Balanced Growth Segment", "Mid-range models showing steady 5G transition."
            
        insights[c] = {
            "label": lbl,
            "note": note,
            "n": len(c_data),
            "avg_rev": avg_rev,
            "avg_units": avg_units,
            "pct_5g": pct_5g
        }
        
    return res_df, len(unique_clusters), labels, sil, explained_var, insights
