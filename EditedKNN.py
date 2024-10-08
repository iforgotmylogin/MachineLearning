import math
from collections import Counter

class EditedKNN:
    def __init__(self, k=3, error_threshold=0.0):
        self.k = k
        self.error_threshold = error_threshold
        self.training_set = []

    def euclidean_distance(self, sample1, sample2):
        distance = 0
        for i in range(len(sample1) - 1):  # Exclude the label
            distance += (sample1[i] - sample2[i]) ** 2
        return math.sqrt(distance)

    def knn_predict(self, test_instance):
        distances = []
        for train_instance in self.training_set:
            dist = self.euclidean_distance(train_instance, test_instance)
            distances.append((train_instance, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]

        # Get the predicted values from neighbors
        output_values = [instance[0][-1] for instance in neighbors]

        # Calculate the predicted value (mean for regression)
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
            # Check if the prediction is within the error threshold
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