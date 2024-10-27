import math
from collections import Counter

class KNNClassifier:
    def __init__(self, k, bandwidth=1.0, distance_metric='euclidean'):
        self.k = k
        self.bandwidth = bandwidth
        self.distance_metric = distance_metric
        self.training_set = []

    def euclidean_distance(self, sample1, sample2):
        distance = 0
        for i in range(len(sample1) - 1):  # Exclude the label
            distance += (sample1[i] - sample2[i]) ** 2
        return math.sqrt(distance)

    def manhattan_distance(self, sample1, sample2):
        distance = 0
        for i in range(len(sample1) - 1):  # Exclude the label
            distance += abs(sample1[i] - sample2[i])
        return distance

    def calculate_distance(self, sample1, sample2):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(sample1, sample2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(sample1, sample2)
        else:
            raise ValueError("Unsupported distance metric")

    # Use Gaussian/RBF kernel for regression
    def gaussian_kernel(self, distance):
        return (1 / (self.bandwidth * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (distance / self.bandwidth) ** 2)

    def knn_predict(self, test_instance):
        distances = []
        for train_instance in self.training_set:
            dist = self.euclidean_distance(train_instance, test_instance)
            distances.append((train_instance, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]

        # For regression, use RBF kernel weights
        weighted_sum = 0.0
        weight_sum = 0.0

        for neighbor, dist in neighbors:
            weight = self.gaussian_kernel(dist)
            weighted_sum += neighbor[-1] * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0  # Prevent division by zero

        return weighted_sum / weight_sum

    def fit(self, training_set):
        self.training_set = training_set

    def predict(self, test_set):
        predictions = []
        for instance in test_set:
            prediction = self.knn_predict(instance)
            predictions.append(prediction)
        return predictions
