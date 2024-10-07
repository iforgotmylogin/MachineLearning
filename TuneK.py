import numpy as np
from KNNClassifier import KNNClassifier
from PreProcessor import PreProcessor

class TuneK:
    def tune(initial_k, DataPath):
        
        label_index = -1 #change here and in main

        preProcessor = PreProcessor()

        # Set the data path
        preProcessor.setDatabase(DataPath)
        
        # Import raw data
        rawData = preProcessor.importData()

        # Clean the data
        cleanedData = preProcessor.cleanData(rawData)

        # Perform stratified split to get class data
        #use for classification
        #classDict, posCount, negCount, neutralCount, otherCount = preProccesor.stratifiedSplit(cleanedData, label_index=10)  # Assuming class label is at index 10

        #use for regressionS
        classDict = preProcessor.regSplit(cleanedData, label_index=7 )  # Assuming class label is at index 10

        # Create folds for cross-validation
        folds = preProcessor.createFolds(classDict, num_folds=10)

        # List to hold accuracies of each fold
        accuracies = []

        bestAccuracy = 0  # Sets a base accuracy of 0
        bestK = initial_k

        for k in range(initial_k, initial_k + 10):  # Adjust this range as needed
            fold_accuracies = []

            for i in range(preProcessor.num_folds):  # Use all folds for cross-validation
                test_fold = folds[i]
                train_folds = [folds[j] for j in range(preProcessor.num_folds) if j != i]
                train_data = [sample for fold in train_folds for sample in fold]

                for j, sample in enumerate(train_data):
                    #print(f"Sample {i} length: {len(sample)}")
                    if (len(sample) == 9):
                       del sample[0]
                for l, sample in enumerate(test_fold):
                    #print(f"Sample {i} length: {len(sample)}")
                    if (len(sample) == 9):
                        del sample[0]


                # Convert training and testing data to NumPy arrays
                X_train = np.array([sample[:label_index] + sample[label_index + 1:] for sample in train_data])  # Features
                y_train = np.array([sample[label_index] for sample in train_data])   # Labels
                X_test = np.array([sample[:label_index] + sample[label_index + 1:] for sample in test_fold])    # Test features
                y_test = np.array([sample[label_index] for sample in test_fold])     # Test labels

                # Predict using the KNNClassifier with the current value of k
                y_pred = [KNNClassifier.predict_classification(X_train, y_train, test_instance, k) for test_instance in X_test]

                # Calculate accuracy of the current fold
                accuracy = np.mean(np.array(y_pred) == y_test)
                fold_accuracies.append(accuracy)

                # Output accuracy for the current fold
                print(f"k = {k} Fold {i + 1} Accuracy: {accuracy}")

            # Calculate average accuracy for this k
            avg_accuracy = np.mean(fold_accuracies)
            accuracies.append(avg_accuracy)
            print(f"k = {k} Fold {i + 1} average Accuracy: {avg_accuracy}")

            # Update the best accuracy and k if necessary
            if avg_accuracy > bestAccuracy:
                bestAccuracy = avg_accuracy
                bestK = k
                print("uptated")
                print(k)

        # Return the value of the best k
        print("The tuned k is " + str(bestK))
        return bestK