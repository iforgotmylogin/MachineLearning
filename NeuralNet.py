import numpy as np

def sigmoid(x):
    # Clip the input to a specified range
    x = np.clip(x, -500, 500)  # Adjust bounds as necessary
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
        input_size = len(data[0][0])

        # Reduced initial weights range
        weights.append(np.random.uniform(-0.5, 0.5, (numHNodes, input_size)))

        for _ in range(1, numHLayers):
            weights.append(np.random.uniform(-0.5, 0.5, (numHNodes, numHNodes)))

        weights.append(np.random.uniform(-0.5, 0.5, (numOutputs, numHNodes)))
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

    def backProp(self, results, label, data, initial_learning_rate=0.001, decay_rate=0.1, epoch=1, gradient_clip_value=0.5):
        learning_rate = initial_learning_rate * np.exp(-decay_rate * epoch)
        all_squared_errors = []
        
        for j, fold in enumerate(data):
            for i, sample in enumerate(fold):
                true_label = sample[label]
                expected = [0] * len(results[i])
                
                # Define expected output based on the true label
                # Code remains the same

                output = results[i]
                squared_error = [(output_val - expected_val) ** 2 for output_val, expected_val in zip(output, expected)]
                all_squared_errors.append(sum(squared_error))

                sample_derivative = [
                    np.clip((output_val - expected_val) * sigmoid_derivative(output_val), -gradient_clip_value, gradient_clip_value)
                    for output_val, expected_val in zip(output, expected)
                ]
                
                # Backpropagate with tighter gradient clipping
                layer_errors = sample_derivative
                for layer_idx in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[layer_idx]
                    new_errors = []

                    for neuron_idx, neuron in enumerate(layer):
                        gradient = np.clip(layer_errors[neuron_idx], -gradient_clip_value, gradient_clip_value)
                        neuron.weights -= learning_rate * gradient * neuron.inputs
                        neuron.bias -= learning_rate * gradient

                        if layer_idx > 0:
                            prev_layer = self.layers[layer_idx - 1]
                            for prev_neuron_idx, prev_neuron in enumerate(prev_layer):
                                new_errors.append([
                                    np.clip(gradient * neuron.weights[prev_neuron_idx], -gradient_clip_value, gradient_clip_value)
                                    for prev_neuron_idx in range(len(neuron.weights))
                                ])

                    if layer_idx > 0:
                        layer_errors = [sum(error) for error in zip(*new_errors)]

        mean_squared_error = np.mean(all_squared_errors)
        print("Mean Squared Error:", mean_squared_error)
        return mean_squared_error

