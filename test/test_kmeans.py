import pytest
import numpy as np
from cluster.kmeans import KMeans 
from cluster.utils import make_clusters

# generating small data set 
X, y = make_clusters(n=10, m=2, k=3, seed=42) # seed for reproducibility 


def test_kmeans_fit():
    """test if KMeans correctly assigns clusters"""
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    assert kmeans.centroids.shape == (3, 2)  # 3 centroids in 2D space
    assert len(kmeans.labels) == len(X)  # each point should have a label

    # check if predicted labels match true labels exactly
    pred_labels = kmeans.predict(X)
    assert np.array_equal(np.sort(np.unique(pred_labels)), np.sort(np.unique(y))), "Predicted labels do not match true labels"

def test_kmeans_predict():
    """test KMeans prediction"""
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    pred_labels = kmeans.predict(X)
    assert pred_labels.shape == (len(X),)  # should return a 1D array
    assert np.all(np.isin(pred_labels, [0, 1, 2]))  # only valid cluster labels

def test_kmeans_centroids():
    """test retrieval of centroids"""
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    centroids = kmeans.get_centroids()
    assert centroids.shape == (3, 2)  # 3 centroids in 2D space

def test_kmeans_error():
    """test calculation of squared-mean error"""
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    error = kmeans.get_error()
    assert isinstance(error, float)  # should return a float
    assert error >= 0  # ecrror should be non-negative