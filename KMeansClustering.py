import numpy as np
import random

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def fit(self, data):
        # Ensure data is a NumPy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Randomly initialize the centroids by sampling from the data
        self.centroids = random.sample(list(data), self.k)  # Convert to list for sampling
        
        # Continue with the fitting process
        # You would add code here for updating centroids, etc.

    def predict(self, data):
        # Logic to assign each data point to the nearest centroid
        predictions = []
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            predictions.append(np.argmin(distances))  # Index of the closest centroid
        return predictions

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))