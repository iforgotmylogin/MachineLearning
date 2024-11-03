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

def setup_preprocessing(data_path):
    preProcessor = PreProcessor()
    preProcessor.reset()
    preProcessor.setDatabase(data_path)
    rawData = preProcessor.importData()
    cleanedData = preProcessor.cleanData(rawData)
    return cleanedData, preProcessor

def train_network(folds, num_output, label_index):
    network = NeuralNet(folds,3, 8, num_output)  # | data | number of hidden layers | number of nodes per hidden layer | number out outputs |
    epoch = 0
    error = 2
    newerror = 1
    max_epochs = 250  # Set maximum epoch limit

    while abs(error - newerror) > 1e-9 and epoch < max_epochs:
        error = newerror
        newerror = network.backProp(network.feedforwardEpoch(folds), label_index, folds, epoch=epoch)
        epoch += 1

    print(f"Training completed in {epoch} epochs.")
    return network

def evaluate_network(network, data, label_index):
    results = network.backProp(network.feedforwardEpoch(data), label_index, data, 1)
    return results

def main():
    data_path = "data/glass.data" # data set 
    numOutput = 2
    label_index = -1

    # Set up preprocessing
    cleaned_data, preProcessor = setup_preprocessing(data_path)

    # Perform stratified split to get class data for classification sets
    classDict, posCount, negCount, neutralCount, otherCount = preProcessor.stratifiedSplit(cleaned_data, label_index)
    folds = preProcessor.createFolds(classDict, num_folds=10)

    results = []
    # Outer loop for testing each fold
    for i, outer_fold in enumerate(folds):
        # Combine all folds except the current outer fold for training
        training_folds = []
        for j, fold in enumerate(folds):
            if j != i:  # Append all folds except the outer fold
                training_folds.extend(fold)

        # Train the neural network on the combined training folds
        network = train_network(training_folds, numOutput, label_index)

        # Evaluate performance on the current outer fold
        fold_performance = evaluate_network(network, outer_fold, label_index)
        results.append(fold_performance)
        print(f'Performance for fold {i+1}: {fold_performance}')  # Print performance for each fold
    avgResult = 0
    for result in results:
        avgResult += result
    avgResult /= len(results)
    print(avgResult)
        

if __name__ == "__main__":
    main()