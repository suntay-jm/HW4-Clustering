import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?
        """
        if isinstance(k, int) and k > 0:
            self.k = k
            self.tol = tol
            self.max_iter = max_iter
            self.centroids = None  # updated during fit
            self.labels = []  # store which cluster each data point belongs to
            self.error = None  # updated during fit
        else:
            raise ValueError("k must be a positive integer")

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        """
        k = self.k
        self.mat = mat
        
        # initializing centroids
        self.centroids = mat[np.random.choice(mat.shape[0], k, replace=False), :]
        
        # clearing so old labels are not accumulated
        self.labels = []

        # assigning points to the nearest centroid
        for point in mat:
            distances = []  # storing all distances of point from centroid
            for centroid in self.centroids:
                distance = np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)  # euclidean distance
                distances.append(distance)
            
            # getting index of smallest distance and assigning it as the label of the nearest centroid
            self.labels.append(np.argmin(distances))

        iter_count = 0

        while iter_count < self.max_iter:
            old_centroids = self.centroids.copy()  # storing old centroids
            new_centroids = []

            for i in range(k):  # looping over k centroids
                cluster_points = mat[np.array(self.labels) == i]  # collecting all points assigned to centroid i

                if cluster_points.shape[0] > 0:  # check to avoid division by 0
                    new_centroid = np.mean(cluster_points, axis=0)
                    new_centroids.append(new_centroid)
                else:  # if no points assigned, reinitialize
                    new_centroids.append(mat[np.random.choice(mat.shape[0], 1), :][0])

            self.centroids = np.array(new_centroids)  # updating centroids

            # reassigning labels after updating centroids
            self.labels = []

            for point in mat:
                distances = []  # storing all distances of point from centroid
                for centroid in self.centroids:
                    distance = np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)  # euclidean distance
                    distances.append(distance)
                
                # getting index of smallest distance and assigning it as the label of the nearest centroid
                self.labels.append(np.argmin(distances))

            # checking for convergence
            new_distance = 0

            for old_centroid, new_centroid in zip(old_centroids, self.centroids):
                new_distance += np.sqrt((old_centroid[0] - new_centroid[0])**2 + (old_centroid[1] - new_centroid[1])**2)

            if new_distance < self.tol:
                break

            iter_count += 1

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points.
        """
        if not isinstance(mat, np.ndarray):
            raise ValueError("mat must be a numpy array")
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D numpy array")
        if self.centroids is None:
            raise ValueError("centroids have not been fitted. call fit() first")
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("mat must have the same number of columns as the centroids")

        pred_labels = []
        for point in mat:
            distances = []
            for centroid in self.centroids:
                distances.append(np.linalg.norm(point - centroid))  # handling cases with more than 2 features
            pred_labels.append(np.argmin(distances))
        return np.array(pred_labels)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model.
        """
        if self.centroids is None:
            raise ValueError("centroids have not been fitted. call fit() first")

        total_error = 0

        for index, point in enumerate(self.mat):
            centroid = self.centroids[self.labels[index]]  # retrieving assigned centroid
            distance = np.linalg.norm(point - centroid) ** 2
            total_error += distance
        return total_error / len(self.mat)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.
        """
        if self.centroids is None:
            raise ValueError("centroids have not been fitted. call fit() first")
        
        return self.centroids
