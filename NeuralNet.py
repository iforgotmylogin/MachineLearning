import numpy as np

def sigmoid(x):
    # Clip the input to a specified range
    #x = np.clip(x, -500, 500)  # Adjust bounds as necessary
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)  # Derivative of the sigmoid function

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.inputs = None  # Store inputs for weight update

    def feedforward(self, inputs):
        self.inputs = inputs  # Store inputs
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class NeuralNet:
    def __init__(self, data, numHLayers, numHNodes, numOutputs): 
        self.layers = []
        weights = NeuralNet.initWeights(data, numHLayers, numHNodes, numOutputs)
        for layer_weights in weights:
            neurons = [Neuron(w, np.random.rand()) for w in layer_weights]
            self.layers.append(neurons)

    @staticmethod
    def initWeights(data, numHLayers, numHNodes, numOutputs):
        weights = []
        input_size = len(data[0][0])  # Number of input features
        weights.append(np.random.rand(numHNodes, input_size))  # Weights for the input layer

        for _ in range(1, numHLayers):
            weights.append(np.random.rand(numHNodes, numHNodes))  # Weights for hidden layers

        weights.append(np.random.rand(numOutputs, numHNodes))  # Weights for the output layer
        return weights

    def feedforwardEpoch(self, data):
        results = []
        for fold in data:
            for sample in fold:
                x = np.array(sample)
                for layer in self.layers:
                    x = np.array([neuron.feedforward(x) for neuron in layer])
                results.append(x)
        return results

    def backProp(self, results, label, data, initial_learning_rate=0.01, decay_rate=0.1, epoch=1):
        num_samples = len(results)
        all_squared_errors = []
        partial_derivatives = []

        # Calculate the learning rate based on the epoch
        learning_rate = initial_learning_rate * np.exp(-decay_rate * epoch)

        for j, fold in enumerate(data):
            for i, sample in enumerate(fold):
                true_label = sample[label]
                expected = [0] * len(results[i])
                expected[int(true_label)] = 1
                output = results[i]
                
                # Calculate errors and store squared errors
                squared_error = [(output_val - expected_val) ** 2 for output_val, expected_val in zip(output, expected)]
                all_squared_errors.append(sum(squared_error))

                # Calculate partial derivatives for each output neuron
                sample_derivative = [(output_val - expected_val) * sigmoid_derivative(output_val) 
                                     for output_val, expected_val in zip(output, expected)]
                partial_derivatives.append(sample_derivative)

                # Backpropagate for each layer
                layer_errors = sample_derivative
                for layer_idx in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[layer_idx]
                    new_errors = []

                    for neuron_idx, neuron in enumerate(layer):
                        gradient = layer_errors[neuron_idx]
                        neuron.weights -= learning_rate * gradient * neuron.inputs
                        neuron.bias -= learning_rate * gradient

                        # Propagate error to previous layer if it's not the input layer
                        if layer_idx > 0:
                            prev_layer = self.layers[layer_idx - 1]
                            for prev_neuron_idx, prev_neuron in enumerate(prev_layer):
                               new_errors.append([gradient * neuron.weights[prev_neuron_idx] for prev_neuron_idx in range(len(neuron.weights))])

                    if layer_idx > 0:
                        layer_errors = [sum(error) for error in zip(*new_errors)]

        mean_squared_error = np.mean(all_squared_errors)
        print("Mean Squared Error:", mean_squared_error)
        return mean_squared_error