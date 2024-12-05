import numpy as np
import matplotlib.pyplot as plt
from PreProcessor import PreProcessor
from NeuralNet import NeuralNet
from PSwarm import PSwarm

def setup_preprocessing(data_path):
    preProcessor = PreProcessor()
    preProcessor.reset()
    preProcessor.setDatabase(data_path)
    rawData = preProcessor.importData()
    cleanedData = preProcessor.cleanData(rawData)
    return cleanedData, preProcessor

def main():
    #data_path = "data/breast-cancer-wisconsin.data"  # Classification data set 
    data_path = "data/glass.data"  # Classification data set 
    #data_path = "data/soybean-small.data"  # Classification data set 

    #data_path = "data/abalone.data"  # Regression data set 
    #data_path = "data/machine.data"  # Regression data set 
    #data_path = "data/forestfires.data"  # Regression data set 

    numOutput = 4  # Adjust based on regression/classification needs
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
        print(f"fold {i+1}")
        # Combine all folds except the current outer fold for training
        training_folds = []
        for j, fold in enumerate(folds):
            if j != i:  # Append all folds except the outer fold
                training_folds.extend(fold)

       # Train the network using Particle Swarm Optimization
        network, _ = PSwarm.train_network(training_folds, numOutput, label_index, is_classification=True)

        # Evaluate performance on the current outer fold using the trained network
        print(outer_fold)
        fold_performance = PSwarm.evaluate_network(network, outer_fold, label_index, is_classification=True) 

        results.append(fold_performance)
        print(f'Error for fold {i+1}: {fold_performance}')  # Print performance for each fold

    avgResult = np.mean(results)  # Calculate average performance
    print(f'Average Error: {avgResult}')  # Print average performance

if __name__ == "__main__":
    main()