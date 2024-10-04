import numpy as np
from KNNClassifier import KNNClassifier
from PreProcessor import PreProcessor

def main():
    preProccesor = PreProcessor()     

    preProccesor.setDatabase("data/breast-cancer-wisconsin.data")
    
    # Import raw data
    rawData = preProccesor.importData()

    # Clean data
    cleanedData = preProccesor.cleanData(rawData)

    # Perform stratified split to get class data
    classDict, posCount, negCount, neutralCount, otherCount = preProccesor.stratifiedSplit(cleanedData, label_index=10)  # Assuming class label is at index 10

    # Create folds from the stratified data
    folds = preProccesor.createFolds(classDict, num_folds=10)  # Create folds for cross-validation

    accuracies = []

    k = 4 # TODO need to tune this later

    for i in range(preProccesor.num_folds): #-1 to leave the 11th fold for tunning 
        test_fold = folds[i]
        train_folds = [folds[j] for j in range(preProccesor.num_folds) if j != i]
        train_data = [sample for fold in train_folds for sample in fold]

        # Convert to NumPy arrays
        X_train = np.array([sample[:10] for sample in train_data])
        y_train = np.array([sample[10] for sample in train_data])
        X_test = np.array([sample[:10] for sample in test_fold])
        y_test = np.array([sample[10] for sample in test_fold])
        
        # Predict on the test fold
        y_pred = [KNNClassifier.predict_classification(X_train,y_train,test_instance,k)for test_instance in X_test]
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        print(f"Fold {i + 1} Accuracy: {accuracy}")

    # Output the average accuracy across all folds
    print(f"Average Accuracy: {np.mean(accuracies)}")

if __name__ == "__main__":
    main()
