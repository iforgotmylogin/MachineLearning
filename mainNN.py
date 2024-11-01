import numpy as np
import matplotlib.pyplot as plt
from PreProcessor import PreProcessor
from NeuralNet import NeuralNet

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

def evaluate(network, test_set, label_index):
    results = network.feedforwardEpoch(test_set)
    all_squared_errors = []

    for i, sample in enumerate(test_set):
        true_label = sample[label_index]
        
        # Ensure results[i] is iterable
        if not isinstance(results[i], (list, np.ndarray)):
            raise TypeError(f"Expected list or array, got {type(results[i])} for results[{i}].")

        expected = [0] * len(results[i])  # Assuming results[i] has multiple values
        expected[int(true_label)] = 1  # Assuming labels are integers representing classes

        output = results[i]
        squared_error = [(output_val - expected_val) ** 2 for output_val, expected_val in zip(output, expected)]
        all_squared_errors.append(sum(squared_error))

    mean_squared_error = np.mean(all_squared_errors)
    return mean_squared_error

def main():
    preProcessor = PreProcessor()
    dataPath = "data/glass.data"
    numOutput = 2
    label_index = -1

    preProcessor.reset()
    preProcessor.setDatabase(dataPath)

    # Import raw data
    rawData = preProcessor.importData()

    # Clean data
    cleanedData = preProcessor.cleanData(rawData)

    # Perform stratified split to get class data for classification sets
    classDict, posCount, negCount, neutralCount, otherCount = preProcessor.stratifiedSplit(cleanedData, label_index)

    # Create folds for cross-validation
    folds = preProcessor.createFolds(classDict, num_folds=10)
    fold_errors = []

    for fold_index in range(10):
        # Separate the folds into training and test sets
        test_set = folds[fold_index]
        training_set = [sample for i, fold in enumerate(folds) if i != fold_index for sample in fold]
        label_index = (len(training_set[0])-1)

        # Initialize a new network for each fold
        network = NeuralNet(training_set, 2, 5, numOutput)  # data, number of hidden layers, number of nodes in each hidden layer, number of outputs(classes)

        # Train the network
        epoch = 0
        error = 2
        newerror = 1
        max_epochs = 250  # Set maximum epoch limit

        while abs(error - newerror) > 1e-5 and epoch < max_epochs:
            error = newerror
            newerror = network.backProp(network.feedforwardEpoch(training_set), label_index, training_set, epoch=1)
            epoch += 1

        print(f"Fold {fold_index + 1}: Training completed in {epoch} epochs.")

        # Evaluate on the test set for the current fold
        fold_error = evaluate(network, test_set, label_index)
        fold_errors.append(fold_error)
        print(f"Fold {fold_index + 1} Test Set Mean Squared Error: {fold_error}")

    # Calculate and display the average error across all folds
    avg_error = np.mean(fold_errors)
    print(f"10-Fold Cross-Validation Mean Squared Error: {avg_error}")

if __name__ == "__main__":
    main()
