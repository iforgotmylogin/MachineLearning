import math
from collections import Counter

class EditedKNN:
    def __init__(self, k=3, error_threshold=0.0, distance_metric='euclidean'):
        self.k = k
        self.error_threshold = error_threshold
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

    def knn_predict(self, test_instance):
        distances = []
        for train_instance in self.training_set:
            dist = self.calculate_distance(train_instance, test_instance)
            distances.append((train_instance, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]

        # Get the predicted values from neighbors
        output_values = [instance[0][-1] for instance in neighbors]

        # Calculate the predicted value
        prediction = sum(output_values) / len(output_values)
        return prediction

    def fit(self, training_set):
        self.training_set = self.edited_knn(training_set)

    def edited_knn(self, training_set):
        edited_set = []
        self.training_set = training_set  # Temporarily set training set for knn_predict
        for instance in training_set:
            temp_set = [x for x in training_set if x != instance]
            self.training_set = temp_set  # Use temp set without the current instance
            prediction = self.knn_predict(instance)
            # Checking if the prediction is in the eror threshold
            if abs(prediction - instance[-1]) <= self.error_threshold:
                edited_set.append(instance)
        self.training_set = edited_set  # Finalize the edited set as training set
        return edited_set

    def predict(self, test_set):
        predictions = []
        for instance in test_set:
            prediction = self.knn_predict(instance)
            predictions.append(prediction)
        return predictions
