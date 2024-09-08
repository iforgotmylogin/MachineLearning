import numpy as np
from NaiveBayes import NaiveBayes
from PreProccessor import PreProccesor

def main():
    preProccesor = PreProccesor()

    preProccesor.setDatabase("MachineLearning/data/breast-cancer-wisconsin.data")
    #preProccesor.createFolds(10)

    rawPos, rawNeg, posCount, negCount = preProccesor.importData()

    folds = preProccesor.createFolds(rawPos, rawNeg, posCount, negCount, num_folds=10) #create folds for cross validation

    accuracies = []

    for i in range(preProccesor.num_folds):
        test_fold = folds[i]
        train_folds = [folds[j] for j in range(preProccesor.num_folds) if j != i]
        train_data = [sample for fold in train_folds for sample in fold]
        
        # Convert to NumPy arrays
        X_train = np.array([sample[:10] for sample in train_data])
        y_train = np.array([sample[10] for sample in train_data])
        X_test = np.array([sample[:10] for sample in test_fold])
        y_test = np.array([sample[10] for sample in test_fold])
        
        # Train the Naive Bayes model
        model = NaiveBayes()
        model.fit(X_train, y_train)
        
        # Predict on the test fold
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        print(f"Fold {i + 1} Accuracy: {accuracy}")

    # Output the average accuracy across all folds
    print(f"Average Accuracy: {np.mean(accuracies)}")

if __name__ == "__main__":
    main()
