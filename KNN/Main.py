import numpy as np
import matplotlib.pyplot as plt
from PreProcessor import PreProcessor
from KNN.EditedKNN import EditedKNN
from KNN.KNNClassifier import KNNClassifier
from KNN.KMeansClustering import KMeansClustering

def plot_accuracies(euclidean_accuracies, manhattan_accuracies):
    methods = ['KNN', 'Edited KNN', 'K-Means']
    x = np.arange(len(methods))

    euclidean_means = [np.mean(acc[:10]) for acc in euclidean_accuracies]
    manhattan_means = [np.mean(acc[10:]) for acc in manhattan_accuracies]

    fig, ax = plt.subplots()
    bar_width = 0.35

    bars1 = ax.bar(x - bar_width / 2, euclidean_means, bar_width, label='Euclidean')
    bars2 = ax.bar(x + bar_width / 2, manhattan_means, bar_width, label='Manhattan')

    ax.set_xlabel('Methods')
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Comparison of Average Accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.show()


def main():
    preProcessor = PreProcessor()

    dataPath = "data/breast-cancer-wisconsin.data"
    label_index = -1

    preProcessor.reset()
    preProcessor.setDatabase(dataPath)

    # Import raw data
    rawData = preProcessor.importData()

    # Clean data
    cleanedData = preProcessor.cleanData(rawData)

    # Perform stratified split to get class data for classification sets
    classDict, posCount, negCount, neutralCount, otherCount = preProcessor.stratifiedSplit(cleanedData, label_index)

    print("main -------------------------------------")

    # Create folds from the stratified data
    folds = preProcessor.createFolds(classDict, num_folds=10)

    knnAccuracies = []
    editedKnnAccuracies = []
    kmeansAccuracies = []

    tuner = TuneK()
    k = tuner.tune(1, dataPath)  # Tunes k on the basic KNN then applies to all the other models

    is_regression = False

    for distance_metric in ['euclidean', 'manhattan']:
        if distance_metric == 'manhattan':
            print("Using Manhattan")

        for i in range(preProcessor.num_folds):
            testFold = folds[i]
            trainFolds = [folds[j] for j in range(preProcessor.num_folds) if j != i]
            trainData = [sample for fold in trainFolds for sample in fold]

            # Remove the first element from samples if they have a length of 9
            for j, sample in enumerate(trainData):
                if len(sample) == 9:
                    del sample[0]
            for l, sample in enumerate(testFold):
                if len(sample) == 9:
                    del sample[0]

            # Convert to NumPy arrays
            X_train = np.array([sample[:label_index] + sample[label_index + 1:] for sample in trainData])  # Features
            y_train = np.array([sample[label_index] for sample in trainData])  # Labels
            X_test = np.array([sample[:label_index] + sample[label_index + 1:] for sample in testFold])  # Test features
            y_test = np.array([sample[label_index] for sample in testFold])  # Test labels

            knn_classifier = KNNClassifier(k=k, distance_metric=distance_metric)
            knn_classifier.fit(trainData)  # Fit the model

            if is_regression:
                # Use KNN for regression
                knnPredictions = knn_classifier.predict(X_test)  # Using features only for prediction
                knnAccuracy = np.mean((knnPredictions - y_test) ** 2)  # Mean Squared Error
                knnAccuracies.append(knnAccuracy)
                print(f"KNN Fold {i + 1} Regression Accuracy ({distance_metric.capitalize()}): {knnAccuracy:.2f}")
            else:
                # Use KNN for classification
                knnPredictions = knn_classifier.predict(testFold)  # Use the full test instance with labels
                knnAccuracy = np.mean(knnPredictions == y_test)
                knnAccuracies.append(knnAccuracy)
                print(f"KNN Fold {i + 1} Classification Accuracy ({distance_metric.capitalize()}): {knnAccuracy:.2f}")



            # Edited KNN Classification
            editedKnnClassifier = EditedKNN(k=k, distance_metric=distance_metric)
            editedKnnClassifier.fit(trainData)
            editedKnnPredictions = editedKnnClassifier.predict(testFold)
            editedKnnAccuracy = np.mean(editedKnnPredictions == y_test)
            editedKnnAccuracies.append(editedKnnAccuracy)
            print(f"Edited KNN Fold {i + 1} Accuracy ({distance_metric.capitalize()}): {editedKnnAccuracy:.2f}")


            # K-Means Clustering
            kMeans = KMeansClustering(k=2, distance_metric=distance_metric)
            kMeans.fit(X_train)

            # Get predictions for the test data
            kMeansTestLabels = kMeans.predict(X_test)

            # Calculate accuracy
            kMeansAccuracy = np.mean(kMeansTestLabels == y_test)
            kmeansAccuracies.append(kMeansAccuracy)
            print(f"K-Means Fold {i + 1} Accuracy ({distance_metric.capitalize()}): {kMeansAccuracy:.2f}")



    # Output the average accuracy across all folds
    if is_regression:
        print(f"Average KNN Regression Accuracy (Euclidean): {np.mean(knnAccuracies[:10]):.2f}")
        print(f"Average KNN Regression Accuracy (Manhattan): {np.mean(knnAccuracies[10:]):.2f}")
    else:
        print(f"Average KNN Classification Accuracy (Euclidean): {np.mean(knnAccuracies[:10]):.2f}")
        print(f"Average KNN Classification Accuracy (Manhattan): {np.mean(knnAccuracies[10:]):.2f}")
    print(f"Average Edited KNN Accuracy (Euclidean): {np.mean(editedKnnAccuracies[:10]):.2f}")
    print(f"Average K-Means Accuracy (Euclidean): {np.mean(kmeansAccuracies[:10]):.2f}")
    print(f"Average Edited KNN Accuracy (Manhattan): {np.mean(editedKnnAccuracies[10:]):.2f}")
    print(f"Average K-Means Accuracy (Manhattan): {np.mean(kmeansAccuracies[10:]):.2f}")

    # Plot the average accuracies
    plot_accuracies([knnAccuracies, editedKnnAccuracies, kmeansAccuracies],
                    [knnAccuracies, editedKnnAccuracies, kmeansAccuracies])


if __name__ == "__main__":
    main()
