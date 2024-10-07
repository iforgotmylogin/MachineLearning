import numpy as np
from KNNClassifier import KNNClassifier
from PreProcessor import PreProcessor
from TuneK import TuneK

def main():
    preProccesor = PreProcessor()     

    dataPath = "data/forestfires.data"
    label_index = -1 #change here and in TuneK

    preProccesor.setDatabase(dataPath)
    
    # Import raw data
    rawData = preProccesor.importData()

    # Clean data
    cleanedData = preProccesor.cleanData(rawData)

    # Perform stratified split to get class data
    #use for classification
    #classDict, posCount, negCount, neutralCount, otherCount = preProccesor.stratifiedSplit(cleanedData, label_index=10)  # Assuming class label is at index 10

    #use for regression
    classDict = preProccesor.regSplit(cleanedData, label_index=label_index )  # Assuming class label is at index 10
    print("main -------------------------------------")
   
    # Create folds from the stratified data
    folds = preProccesor.createFolds(classDict, num_folds=10)  # Create folds for cross-validation

    accuracies = []

    k = TuneK.tune(1, dataPath) #tunes k starting with k =1 and increases by 1 ten times

    for i in range(preProccesor.num_folds): 
        test_fold = folds[i]
        train_folds = [folds[j] for j in range(preProccesor.num_folds) if j != i]
        train_data = [sample for fold in train_folds for sample in fold]
             
        for j, sample in enumerate(train_data):
             #print(f"Sample {i} length: {len(sample)}")
             if (len(sample) == 9):
                del sample[0]
        for l, sample in enumerate(test_fold):
             #print(f"Sample {i} length: {len(sample)}")
             if (len(sample) == 9):
                del sample[0]

       
        # Convert to NumPy arrays
        X_train = np.array([sample[:label_index] + sample[label_index + 1:] for sample in train_data])  # Features
        y_train = np.array([sample[label_index] for sample in train_data])   # Labels
        X_test = np.array([sample[:label_index] + sample[label_index + 1:] for sample in test_fold])    # Test features
        y_test = np.array([sample[label_index] for sample in test_fold])     # Test labels

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
