import numpy as np
import matplotlib.pyplot as plt
from PreProcessor import PreProcessor
from NeuralNet import NeuralNet

def setup_preprocessing(data_path):
    preProcessor = PreProcessor()
    preProcessor.reset()
    preProcessor.setDatabase(data_path)
    rawData = preProcessor.importData()
    cleanedData = preProcessor.cleanData(rawData)
    return cleanedData, preProcessor

def train_network(folds, num_output, label_index, is_classification):
    network = NeuralNet(folds, 2, 5, num_output)  # | data | num hidden layers | number of nodes per layer | number of outputs |
    epoch = 0
    error = 2
    newerror = 1
    max_epochs = 250  # Set maximum epoch limit

    while abs(error - newerror) > 1e-9 and epoch < max_epochs:
        error = newerror
        if is_classification:
            newerror = network.backProp_classification(network.feedforwardEpoch(folds), label_index, folds, epoch=epoch)
        else:
            newerror = network.backProp_regression(network.feedforwardEpoch(folds), label_index, folds, epoch=epoch)
        epoch += 1

    print(f"Training completed in {epoch} epochs.")
    return network

def evaluate_network(network, data, label_index, is_classification):
    if is_classification:
        results = network.backProp_classification(network.feedforwardEpoch(data), label_index, data, 1)
    else:
        results = network.backProp_regression(network.feedforwardEpoch(data), label_index, data, 1)
    return results

def main():
    #data_path = "data/breast-cancer-wisconsin.data"  # Classification data set 
    #data_path = "data/glass.data"  # Classification data set 
    #data_path = "data/soybean-small.data"  # Classification data set 

    #data_path = "data/abalone.data"  # Regression data set 
    #data_path = "data/machine.data"  # Regression data set 
    data_path = "data/forestfires.data"  # Regression data set 

    numOutput = 20  # Adjust based on regression/classification needs
    label_index = -1

    # Set up preprocessing
    cleaned_data, preProcessor = setup_preprocessing(data_path)

    # Perform stratified split to get class data for classification sets
    #classDict, posCount, negCount, neutralCount, otherCount = preProcessor.stratifiedSplit(cleaned_data, label_index)

    #IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Perform stratified split to get class data for Regression sets
    classDict = preProcessor.regSplit(cleaned_data, label_index)

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
        network = train_network(training_folds, numOutput, label_index, is_classification=False)

        # Evaluate performance on the current outer fold
        fold_performance = evaluate_network(network, outer_fold, label_index, is_classification=False)
        results.append(fold_performance)
        print(f'Performance for fold {i+1}: {fold_performance}')  # Print performance for each fold

    avgResult = np.mean(results)  # Calculate average performance
    print(f'Average Performance: {avgResult}')  # Print average performance

if __name__ == "__main__":
    main()
