import numpy as np
from KNN.KNNClassifier import KNNClassifier
from KNN.PreProcessor import PreProcessor

class TuneK:
    def tune(self, initial_k, DataPath):
        
       

        preProcessor = PreProcessor()

        
        # Set the data path
        preProcessor.setDatabase(DataPath)

        label_index = -1
        preProcessor.reset()
        # Import raw data
        rawData = preProcessor.importData()

        # Clean the data
        cleanedData = preProcessor.cleanData(rawData)

        # Perform stratified split to get class data (use appropriate method for your task)
        #classDict = preProcessor.regSplit(cleanedData, label_index=7)  # Assuming class label is at index 7

        # Perform stratified split to get class data for classification sets 
        classDict, posCount, negCount, neutralCount, otherCount = preProcessor.stratifiedSplit(cleanedData, label_index)

        # Create folds for cross-validation
        folds = preProcessor.createFolds(classDict, num_folds=10)

        # List to hold accuracies of each fold
        accuracies = []

        bestAccuracy = 0  # Initialize base accuracy
        bestK = 0

        for k in range(initial_k, initial_k + 10):  # Adjust the range as needed
            fold_accuracies = []

            for i in range(preProcessor.num_folds):  # Loop over all folds
                test_fold = folds[i]
                train_folds = [folds[j] for j in range(preProcessor.num_folds) if j != i]
                train_data = [sample for fold in train_folds for sample in fold]

                # Remove any extra fields if necessary
                for j, sample in enumerate(train_data):
                    if len(sample) == 9:
                        del sample[0]
                for l, sample in enumerate(test_fold):
                    if len(sample) == 9:
                        del sample[0]

                # Convert training and testing data to NumPy arrays
                X_train = np.array([sample[:label_index] + sample[label_index + 1:] for sample in train_data])  # Features
                y_train = np.array([sample[label_index] for sample in train_data])  # Labels
                X_test = np.array([sample[:label_index] + sample[label_index + 1:] for sample in test_fold])  # Test features
                y_test = np.array([sample[label_index] for sample in test_fold])  # Test labels

                # Combine features and labels for training
                training_set = np.column_stack((X_train, y_train))

                # Create the KNN classifier with the current k
                k = k
                knn_classifier = KNNClassifier(k)

                # Fit the classifier with the training set
                knn_classifier.fit(training_set)

                # Predict using the classifier on the test set
                y_pred = knn_classifier.predict(X_test)

                # Calculate accuracy for the current fold
                accuracy = np.mean(np.array(y_pred) == y_test)
                fold_accuracies.append(accuracy)

                # Output accuracy for the current fold
                print(f"k = {k} Fold {i + 1} Accuracy: {accuracy}")

            # Calculate average accuracy for this k
            avg_accuracy = np.mean(fold_accuracies)
            accuracies.append(avg_accuracy)
            print(f"k = {k} Average Accuracy: {avg_accuracy}")

            # Update the best accuracy and k if necessary
            if avg_accuracy > bestAccuracy:
                bestAccuracy = avg_accuracy
                bestK = k
                print(f"Updated best k to: {k} with accuracy: {bestAccuracy}")

        # Return the value of the best k
        print(f"The tuned k is: {bestK}")
        return bestK
