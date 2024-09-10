import numpy as np
from NaiveBayes import NaiveBayes
from PreProccessor import PreProccesor

def main():
    preprocessor = PreProccesor()

    preprocessor.setDatabase("data/breast-cancer-wisconsin.data")

    rawPos, rawNeg, posCount, negCount = preprocessor.importData()
    folds = preprocessor.createFolds(rawPos, rawNeg, posCount, negCount, 10)

    print("Cross-validation without noise:")
    accuracies_without_noise, entropies_without_noise = cross_validate(folds, preprocessor)

    print(f"Average Accuracy without noise: {np.mean(accuracies_without_noise)}")
    print(f"Average Entropy Loss without noise: {np.mean(entropies_without_noise)}")

    # Introduce noise
    preprocessor.generateNoise(folds)

    # Check if noise has been introduced
    print("Data after noise introduction (first fold sample):", folds[0][:5])

    print("Cross-validation with noise:")
    accuracies_with_noise, entropies_with_noise = cross_validate(folds, preprocessor)

    print(f"Average Accuracy with noise: {np.mean(accuracies_with_noise)}")
    print(f"Average Entropy Loss with noise: {np.mean(entropies_with_noise)}")



def cross_validate(folds, preprocessor):
    accuracies = []
    entropys = []

    for i in range(preprocessor.num_folds):
        test_fold = folds[i]
        train_folds = [folds[j] for j in range(preprocessor.num_folds) if j != i]
        train_data = [sample for fold in train_folds for sample in fold]

        X_train = np.array([sample[:10] for sample in train_data])
        y_train = np.array([sample[10] for sample in train_data])
        X_test = np.array([sample[:10] for sample in test_fold])
        y_test = np.array([sample[10] for sample in test_fold])

        model = NaiveBayes()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

          # Calculate entropy loss function
        elf = entropyLoss(y_test,y_pred)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)

        
        accuracies.append(accuracy)
        entropys.append(elf)
        print(f"Fold {i + 1} Entropy loss: {elf}")
        print(f"Fold {i + 1} Accuracy: {accuracy}")

    return accuracies, entropys

def entropyLoss(y_actual, y_predict):
    loss = np.sum(y_actual *np.log(y_predict)/y_actual.shape[0])
    return loss


if __name__ == "__main__":
    main()
