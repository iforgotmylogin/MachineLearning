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


def main():
    preProcessor = PreProcessor()

    dataPath = "data/breast-cancer-wisconsin.data"
    label_index = -1

    preProcessor.reset()
    preProcessor.setDatabase(dataPath)

    # Import raw data
    rawData = preProcessor.importData()

    # Clean data
    cleanedData = preProcessor.cleanData(rawData)

    # Perform stratified split to get class data for classification sets
    classDict, posCount, negCount, neutralCount, otherCount = preProcessor.stratifiedSplit(cleanedData, label_index)

    print("main -------------------------------------")

    # Create folds from the stratified data
    
    folds = preProcessor.createFolds(classDict, num_folds=10)
    
    network = NeuralNet(folds, 2, 4, 2) #data, number of hidden layers, number of nodes in each hidden layer, number of outputs(classes)

    
    network.backProp(network.feedforwardEpoch(folds),label_index,folds)

if __name__ == "__main__":
    main()
