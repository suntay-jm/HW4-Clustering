import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from cluster.utils import make_clusters
from sklearn.metrics import silhouette_samples

# generating a small dataset
X, y = make_clusters(n=10, m=2, k=3, seed=42)  # seed for reproducibility

def test_silhouette_score():
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    pred_labels = kmeans.predict(X)
    
    silhouette = Silhouette()
    scores = silhouette.score(X, pred_labels)
    
    assert scores.shape == (len(X),)  # one score per data point
    assert np.all(scores >= -1) and np.all(scores <= 1)  # scores should be in [-1,1]

def test_silhouette_vs_sklearn():
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    pred_labels = kmeans.predict(X)
    
    silhouette = Silhouette()
    scores = silhouette.score(X, pred_labels)  # mine

    sklearn_scores = silhouette_samples(X, pred_labels)  # sklearn's 

    print("\n--- Silhouette Score Comparison ---")
    for i in range(len(scores)):
        print(f"Index {i}: Mm Score = {scores[i]:.4f}, sklearn Score = {sklearn_scores[i]:.4f}")

    # ensure scores match within 2 decimal places
    np.testing.assert_almost_equal(scores, sklearn_scores, decimal=2)  
