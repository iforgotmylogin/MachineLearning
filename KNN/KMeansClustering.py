import numpy as np
import random

class KMeansClustering:
    def __init__(self, k=3, distance_metric='euclidean', n_init=10, max_iter=300):
        self.k = k
        self.distance_metric = distance_metric
        self.n_init = n_init  # Number of times to run the algorithm with different centroid seeds
        self.max_iter = max_iter  # Maximum number of iterations for a single run
        self.centroids = None

    def fit(self, data):
        # Ensure data is a NumPy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        best_centroids = None
        best_inertia = float('inf')

        for _ in range(self.n_init):
            # Randomly initialize the centroids by sampling from the data
            self.centroids = random.sample(list(data), self.k)
            for _ in range(self.max_iter):
                clusters = self._assign_clusters(data)
                new_centroids = self._update_centroids(data, clusters)
                if np.allclose(self.centroids, new_centroids):
                    break
                self.centroids = new_centroids

            inertia = self._calculate_inertia(data, clusters)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids

        self.centroids = best_centroids

    def predict(self, data):
        predictions = []
        for point in data:
            distances = [self._distance(point, centroid) for centroid in self.centroids]
            predictions.append(np.argmin(distances))  # Index of the closest centroid
        return predictions

    def _distance(self, point1, point2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))

    def _assign_clusters(self, data):
        clusters = [[] for _ in range(self.k)]
        for idx, point in enumerate(data):
            closest_centroid = np.argmin([self._distance(point, centroid) for centroid in self.centroids])
            clusters[closest_centroid].append(idx)
        return clusters

    def _update_centroids(self, data, clusters):
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = np.mean(data[cluster], axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data))
        return new_centroids

    def _calculate_inertia(self, data, clusters):
        inertia = 0.0
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                inertia += self._distance(data[idx], self.centroids[i]) ** 2
        return inertia