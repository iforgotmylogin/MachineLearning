import numpy as np
from PreProcessor import PreProcessor
from EditedKNN import EditedKNN
from KNNClassifier import KNNClassifier
from KMeansClustering import KMeansClustering

def main():
    preProcessor = PreProcessor()

    dataPath = "data/forestfires.data"
    label_index = -1  

    preProcessor.setDatabase(dataPath)

    # Import raw data
    rawData = preProcessor.importData()

    # Clean data
    cleanedData = preProcessor.cleanData(rawData)

    # Perform stratified split to get class data
    classDict = preProcessor.regSplit(cleanedData, label_index=label_index)
    print("main -------------------------------------")

    # Create folds from the stratified data
    folds = preProcessor.createFolds(classDict, num_folds=10)

    knnAccuracies = []
    editedKnnAccuracies = []
    kmeansAccuracies = []

    k = 3  # Example k value for KNN and Edited KNN

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
        y_train = np.array([sample[label_index] for sample in trainData])   # Labels
        X_test = np.array([sample[:label_index] + sample[label_index + 1:] for sample in testFold])    # Test features
        y_test = np.array([sample[label_index] for sample in testFold])     # Test labels

        # KNN Classification
        knn_classifier = KNNClassifier(k=k)  # Create an instance of KNNClassifier
        knn_classifier.fit(trainData)  # Fit the model
        knnPredictions = knn_classifier.predict(testFold)  # Use the predict method
        knnAccuracy = np.mean(knnPredictions == y_test)
        knnAccuracies.append(knnAccuracy)
        print(f"KNN Fold {i + 1} Accuracy: {knnAccuracy:.2f}")

        # Edited KNN Classification
        editedKnnClassifier = EditedKNN(k=k)
        editedKnnClassifier.fit(trainData)
        editedKnnPredictions = editedKnnClassifier.predict(testFold)
        editedKnnAccuracy = np.mean(editedKnnPredictions == y_test)
        editedKnnAccuracies.append(editedKnnAccuracy)
        print(f"Edited KNN Fold {i + 1} Accuracy: {editedKnnAccuracy:.2f}")

        # K-Means Clustering
        kMeans = KMeansClustering(k=2) 
        kMeans.fit(X_train)
        
        # Get predictions for the test data
        kMeansTestLabels = kMeans.predict(X_test)
        
        # Calculate accuracy
        kMeansAccuracy = np.mean(kMeansTestLabels == y_test)
        kmeansAccuracies.append(kMeansAccuracy)
        print(f"K-Means Fold {i + 1} Accuracy: {kMeansAccuracy:.2f}")

    # Output the average accuracy across all folds
    print(f"Average KNN Accuracy: {np.mean(knnAccuracies):.2f}")
    print(f"Average Edited KNN Accuracy: {np.mean(editedKnnAccuracies):.2f}")
    print(f"Average K-Means Accuracy: {np.mean(kmeansAccuracies):.2f}")

if __name__ == "__main__":
    main()
