import numpy as np
from PreProcessor import PreProcessor
from NeuralNet import NeuralNet

def print_weights(network):
    print("Network weights and biases:")
    for layer_idx, layer in enumerate(network.layers):
        print(f"Layer {layer_idx + 1}:")
        for neuron_idx, neuron in enumerate(layer):
            print(f"  Neuron {neuron_idx + 1}: Weights = {neuron.weights}, Bias = {neuron.bias}")


def print_activations(network, sample):
    activations = []
    x = np.array(sample)
    for layer in network.layers:
        x = np.array([neuron.feedforward(x) for neuron in layer])
        activations.append(x)
    print("Activations at each layer:")
    for layer_idx, activation in enumerate(activations):
        print(f"  Layer {layer_idx + 1}: {activation}")


def main():
    preProcessor = PreProcessor()
    dataPath = "data/machine copy.data"
    numOutput = 3
    label_index = -1

    preProcessor.reset()
    preProcessor.setDatabase(dataPath)

    # Import raw data
    rawData = preProcessor.importData()

    # Clean data
    cleanedData = preProcessor.cleanData(rawData)

    # Perform stratified split to get class data for classification sets
    classDict, posCount, negCount, neutralCount, otherCount = preProcessor.stratifiedSplit(cleanedData, label_index)

    folds = preProcessor.createFolds(classDict, num_folds=10)

    mse_per_model = []
    training_times_per_model = []

    for numHLayers in [0, 1, 2]:
        print(f"\nTraining with {numHLayers} hidden layers:")
        network = NeuralNet(folds, numHLayers, 5, numOutput)
        training_time, average_mse = network.train_model(folds, label_index)
        mse_per_model.append(average_mse)
        training_times_per_model.append(training_time)

    print("\nResults Summary:")
    for i, num_layers in enumerate([0, 1, 2]):
        print(f"Model with {num_layers} hidden layers:")
        print(f"  Average MSE: {mse_per_model[i]:.4f}")
        print(f"  Training Time: {training_times_per_model[i]:.2f} seconds")

if __name__ == "__main__":
    main()
