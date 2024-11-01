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
        self.output = None  # Store the output of the neuron

    def feedforward(self, inputs):
        self.inputs = inputs  # Store inputs
        total = np.dot(self.weights, inputs) + self.bias
        self.output = sigmoid(total)  # Store output
        return self.output  # Return the output

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
        input_size = len(data[0])

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

        for i, sample in enumerate(data):
            true_label = data[label]
            expected = [1 if j == true_label else 0 for j in range(len(results[i]))]
            output = results[i]

            squared_error = [(output_val - expected_val) ** 2 for output_val, expected_val in zip(output, expected)]
            all_squared_errors.append(sum(squared_error))

            sample_derivative = [
                np.clip((output_val - expected_val) * sigmoid_derivative(output_val), -gradient_clip_value, gradient_clip_value)
                for output_val, expected_val in zip(output, expected)
            ]

            layer_errors = sample_derivative
            for layer_idx in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_idx]
                new_errors = [0] * (len(self.layers[layer_idx - 1]) if layer_idx > 0 else len(sample_derivative))

                for neuron_idx, neuron in enumerate(layer):
                    gradient = np.clip(layer_errors[neuron_idx], -gradient_clip_value, gradient_clip_value)
                    neuron.weights -= learning_rate * gradient * (self.layers[layer_idx - 1][neuron_idx].output if layer_idx > 0 else sample)
                    neuron.bias -= learning_rate * gradient

                    if layer_idx > 0:
                        for prev_neuron_idx in range(len(new_errors)):
                            new_errors[prev_neuron_idx] += gradient * neuron.weights[prev_neuron_idx]

                layer_errors = new_errors

        mean_squared_error = np.mean(all_squared_errors)
        print("Mean Squared Error:", mean_squared_error)
        return mean_squared_error
