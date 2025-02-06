import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y aren't numpy arrays")
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y don't have the right dimensions and/or don't have the same number of rows")

        silhouette_scores = np.zeros(X.shape[0]) # initializing 'empty' array as same length as X to hold scores for each point

        for i in range(len(X)): # going over every point in X
            # compute a(i) - mean intra-cluster distance
            same_cluster = X[y == y[i]] # getting all points in the same cluster as X[i]
            if len(same_cluster) > 1:  
                distances = cdist(X[i].reshape(1, -1), same_cluster)[0]  # calculating euclidean distance between X[i] and all points in the same cluster
                ai = np.mean(distances[distances > 0])  # exclude self-distance
            else:
                ai = np.nan  # if a cluster has only one point, set as NaN (handle later)

            # compute b(i) - mean nearest-cluster distance
            bi_values = [] # to hold mean distances between X[i] and other clusters
            for label in np.unique(y): 
                if label != y[i]: # skip cluster that X[i] belongs to and ensure only comparing X[i] to other clusters
                    cluster_points = X[y == label] # getting all points that belong to the current cluster
                    distances = cdist(X[i].reshape(1, -1), cluster_points)[0]
                    bi_values.append(np.mean(distances)) # mean between X[i] and each other cluster

            bi = min(bi_values) if bi_values else 0  # pick the smallest mean distance among all other clusters

            # compute silhouette score
            if np.isnan(ai):  # if a(i) is NaN (one cluster), set silhouette score to 0.0
                silhouette_scores[i] = 0.0
            else:
                silhouette_scores[i] = (bi - ai) / max(ai, bi) if max(ai, bi) != 0 else 0 # using silhouette score formula

        return silhouette_scores
