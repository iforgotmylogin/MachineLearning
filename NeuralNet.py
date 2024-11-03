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
        input_size = len(data[0])  # Size of input layer based on sample length

        # Initialize weights for the first hidden layer, with `input_size` inputs
        weights.append(np.random.uniform(-0.5, 0.5, (numHNodes, input_size)))

        # Initialize weights for subsequent hidden layers with `numHNodes` inputs
        for _ in range(1, numHLayers):
            weights.append(np.random.uniform(-0.5, 0.5, (numHNodes, numHNodes)))

        # Initialize weights for the output layer with `numHNodes` inputs
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

    def backProp(self, results, label_index, data, initial_learning_rate=0.001, decay_rate=0.1, epoch=1, gradient_clip_value=0.5):
        learning_rate = initial_learning_rate * np.exp(-decay_rate * epoch)
        all_squared_errors = []

        for i, sample in enumerate(data):
            true_label = sample[label_index]  # Correctly fetch the label from each sample
            expected = [1 if j == true_label else 0 for j in range(len(results[i]))]
            output = results[i]

            # Ensure lengths of expected and output match
            if len(expected) != len(output):
                raise ValueError(f"Output length {len(output)} does not match expected length {len(expected)}")

            squared_error = [(output_val - expected_val) ** 2 for output_val, expected_val in zip(output, expected)]
            all_squared_errors.append(sum(squared_error))

            sample_derivative = [
                np.clip((output_val - expected_val) * sigmoid_derivative(output_val), -gradient_clip_value, gradient_clip_value)
                for output_val, expected_val in zip(output, expected)
            ]

            layer_errors = sample_derivative

            for layer_idx in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_idx]
                new_errors = np.zeros(len(self.layers[layer_idx - 1])) if layer_idx > 0 else np.zeros(len(sample_derivative))

                for neuron_idx, neuron in enumerate(layer):
                    gradient = np.clip(layer_errors[neuron_idx], -gradient_clip_value, gradient_clip_value)

                    # Get inputs based on the layer; first layer uses sample as input
                    if layer_idx == 0:
                        inputs = sample  # First layer uses the sample as input
                    else:
                        # Gather outputs from the previous layer
                        inputs = np.array([prev_neuron.output for prev_neuron in self.layers[layer_idx - 1]])

                    # Debugging statements to check shapes
                    print(f"Layer {layer_idx}, Neuron {neuron_idx}:")
                    print(f"  Input shape: {inputs.shape}, Weight shape: {neuron.weights.shape}")

                    # Check if weights are 2D and adjust the shape checking accordingly
                    if len(neuron.weights.shape) == 1:  # Single neuron with weights
                        if neuron.weights.shape[0] != inputs.shape[0]:
                            raise ValueError(f"Shape mismatch: Weight shape {neuron.weights.shape} and input shape {inputs.shape} do not match.")
                    elif len(neuron.weights.shape) == 2:  # Layer with multiple inputs
                        if neuron.weights.shape[1] != inputs.shape[0]:
                            raise ValueError(f"Shape mismatch: Weight shape {neuron.weights.shape} and input shape {inputs.shape} do not match.")

                    # Update weights and bias
                    neuron.weights -= learning_rate * gradient * inputs
                    neuron.bias -= learning_rate * gradient

                    # Update errors for backpropagation
                    if layer_idx > 0:
                        for prev_neuron_idx in range(len(new_errors)):
                            new_errors[prev_neuron_idx] += gradient * neuron.weights[prev_neuron_idx]

                layer_errors = new_errors

        mean_squared_error = np.mean(all_squared_errors)
        print("Mean Squared Error:", mean_squared_error)
        return mean_squared_error
