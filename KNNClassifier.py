import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def get_neighbors(training_set, training_labels, test_instance, k):
    distances = []
    for i in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[i])
        distances.append((training_labels[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors


def predict_classification(training_set, training_labels, test_instance, k):
    neighbors = get_neighbors(training_set, training_labels, test_instance, k)
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))


def precision_metric(actual, predicted):
    true_positives = sum((pred == 1 and act == 1) for pred, act in zip(predicted, actual))
    predicted_positives = sum(predicted)
    return true_positives / predicted_positives if predicted_positives > 0 else 0


def recall_metric(actual, predicted):
    true_positives = sum((pred == 1 and act == 1) for pred, act in zip(predicted, actual))
    actual_positives = sum(actual)
    return true_positives / actual_positives if actual_positives > 0 else 0


def f1_score_metric(actual, predicted):
    precision = precision_metric(actual, predicted)
    recall = recall_metric(actual, predicted)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
