import numpy as np

def sigmoid(x):
    # Activation Function
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, and then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class NeuralNet:
    def __init__(self, data, numHLayers, numHNodes, numOutputs): 
        # Initialize weights and layers
        self.layers = []
        weights = NeuralNet.initWeights(data, numHLayers, numHNodes, numOutputs)
        
        # Initialize each layer of neurons
        for layer_weights in weights:
            neurons = [Neuron(w, np.random.rand()) for w in layer_weights]  # Random biases
            self.layers.append(neurons)

    @staticmethod
    def initWeights(data, numHLayers, numHNodes, numOutputs):
        # Create weight matrices for hidden layers and the output layer
        weights = []
        input_size = len(data[0][0])  # Number of input features

        # Initialize weights for input to first hidden layer
        weights.append(np.random.rand(numHNodes, input_size))

        # Initialize weights for each hidden layer
        for _ in range(1, numHLayers):
            weights.append(np.random.rand(numHNodes, numHNodes))
        
        # Initialize weights for last hidden layer to output layer
        weights.append(np.random.rand(numOutputs, numHNodes))
        
        return weights

    def feedforwardEpoch(self, data):
        results = []  # Store the results for each sample
        
        for fold in data:
            for sample in fold:
                x = np.array(sample)  # Convert the entire sample to a numpy array
                
                # Pass inputs through each layer
                for layer in self.layers:
                    # For each neuron in the current layer, perform feedforward using the sample as input
                    x = np.array([neuron.feedforward(x) for neuron in layer])
                
                results.append(x)  # Store output for each sample

        print(results)
        return results  # Return all outputs from the outputlayer

