import numpy as np
from NaiveBayes import NaiveBayes
from PreProcessor import PreProcessor

def main():
    preprocessor = PreProcessor()

    # Set the dataset path (adjust this according to your dataset)
    preprocessor.setDatabase("data/glass.data")  # Adjust the path as necessary

    # Import and process the data
    raw_data = preprocessor.importData()
    cleaned_data = preprocessor.cleanData(raw_data)
    
    # Stratify and get class counts
    label_index = -1  # Assuming the last column contains the class labels
    class_dict, posCount, negCount, neutralCount, otherCount = preprocessor.stratifiedSplit(cleaned_data, label_index)

    # Create 10 stratified folds
    folds = preprocessor.createFolds(class_dict, num_folds=10)

    #TODO ----------------------------------------------change after this point to impliment KNN
    
    # Display cross-validation results without noise
    print("Cross-validation without noise:")
    accuracies_without_noise, entropies_without_noise = cross_validate(folds, preprocessor)

    print(f"Average Entropy Loss without noise: {np.mean(entropies_without_noise)}")
    print(f"Average Accuracy without noise: {np.mean(accuracies_without_noise)}")

    # Introduce noise into the dataset
    preprocessor.generateNoise(folds)

    # Display the data after noise introduction (showing the first 5 samples of the first fold)
    print("Data after noise introduction (first fold sample):", folds[0][:5])

    # Display cross-validation results with noise
    print("Cross-validation with noise:")
    accuracies_with_noise, entropies_with_noise = cross_validate(folds, preprocessor)

    print(f"Average Accuracy with noise: {np.mean(accuracies_with_noise)}")
    print(f"Average Entropy Loss with noise: {np.mean(entropies_with_noise)}")

def cross_validate(folds, preprocessor):
    accuracies = []
    entropies = []

    for i in range(preprocessor.num_folds):
        test_fold = folds[i]
        train_folds = [folds[j] for j in range(preprocessor.num_folds) if j != i]
        train_data = [sample for fold in train_folds for sample in fold]

        # Separate features (X) and labels (y) for training and testing data
        X_train = np.array([sample[:-1] for sample in train_data])  # Features (all columns except the last one)
        y_train = np.array([sample[-1] for sample in train_data])   # Labels (last column)
        X_test = np.array([sample[:-1] for sample in test_fold])
        y_test = np.array([sample[-1] for sample in test_fold])

        # Initialize and fit the Naive Bayes model
        model = NaiveBayes()
        model.fit(X_train, y_train)

        # Predict using the model
        y_pred = model.predict(X_test)

        # Calculate entropy loss
        entropy_loss = entropyLoss(y_test, y_pred)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)

        # Append results
        accuracies.append(accuracy)
        entropies.append(entropy_loss)

        # Print individual fold results
        print(f"Fold {i + 1} Entropy loss: {entropy_loss}")
        print(f"Fold {i + 1} Accuracy: {accuracy}")

    return accuracies, entropies

def entropyLoss(y_actual, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Here, adjust y_actual and y_pred shape handling for the loss calculation
    loss = -np.mean(y_actual * np.log(y_pred))
    return loss

if __name__ == "__main__":
    main()
