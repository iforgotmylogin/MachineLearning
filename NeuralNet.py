import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)  # Derivative of the sigmoid function

def linear(x):
    return x  # Linear activation function

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.inputs = None  # Store inputs for weight update
        self.output = None  # Store the output of the neuron

    def feedforward(self, inputs, use_linear=False):
        self.inputs = inputs  # Store inputs for backpropagation
        total = np.dot(self.weights, inputs) + self.bias
        
        if use_linear:
            self.output = linear(total)  # Linear activation for regression
        else:
            self.output = sigmoid(total)  # Sigmoid activation for output
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

        # Initialize weights for the input layer to the first hidden layer
        weights.append(np.random.uniform(-0.5, 0.5, (numHNodes, input_size)))

        # Initialize weights for hidden layers
        for _ in range(1, numHLayers):
            weights.append(np.random.uniform(-0.5, 0.5, (numHNodes, numHNodes)))

        # Initialize weights for the output layer
        weights.append(np.random.uniform(-0.5, 0.5, (numOutputs, numHNodes)))
        
        return weights

    def feedforwardEpoch(self, data):
        results = []
        for sample in data:
            x = np.array(sample)
            for layer in self.layers:
                x = np.array([neuron.feedforward(x) for neuron in layer])
            results.append(x)
        # print("layer outputs")
        # print(results)
        return results
    
    def get_weights(self):
        """Retrieve all weights in the network as a flat list."""
        all_weights = []
        for layer in self.layers:
            for neuron in layer:
                all_weights.append(neuron.weights)
        #print(all_weights)
        return all_weights
    
    def set_weights(self, new_weights):
        """Set new weights for the entire network."""

        #print(new_weights)
        # Flatten the new weights into the structure of each layer
        index = 0
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer):
                # Reshape the new_weights to match the dimensions
                if new_weights[i][j].shape:  # Checks if the shape is not empty
                    neuron.weights = new_weights[i][j]
                else:
                    print("Invalid weights shape detected!")
                    # You can initialize with a default shape and values, e.g., zeroed array of appropriate size
                    #neuron.weights = np.zeros_like(neuron.weights)  # Assuming neuron.weights has an appropriate shape


    def backProp_classification(self, results, label_index, data, initial_learning_rate=0.015, decay_rate=0.1, epoch=1, gradient_clip_value=0.5):
        learning_rate = initial_learning_rate * np.exp(-decay_rate * epoch)
        all_squared_errors = []

        for i, sample in enumerate(data):
            true_label = int(sample[label_index])  # Ensure true_label is an integer
            output = results[i].flatten()  # Ensure the output is flat
            expected = np.zeros_like(output)
            expected[true_label] = 1  # One-hot encoding for classification

            squared_error = (output - expected) ** 2
            all_squared_errors.append(np.sum(squared_error))

            # Calculate the derivative of the output error
            sample_derivative = np.clip((output - expected) * sigmoid_derivative(output), -gradient_clip_value, gradient_clip_value)

            # Backpropagation
            layer_errors = sample_derivative
            for layer_idx in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_idx]
                new_errors = np.zeros((len(self.layers[layer_idx - 1]),)) if layer_idx > 0 else None

                for neuron_idx, neuron in enumerate(layer):
                    #print("Neuron weights shape:", neuron.weights.shape)

                    gradient = layer_errors[neuron_idx]
                    neuron.weights -= learning_rate * gradient * neuron.inputs
                    neuron.bias -= learning_rate * gradient

                    if layer_idx > 0:  # Backpropagate errors
                        for prev_neuron_idx in range(len(neuron.weights)):
                            new_errors[prev_neuron_idx] += gradient * neuron.weights[prev_neuron_idx]

                if layer_idx > 0:
                    layer_errors = new_errors

        mean_squared_error = np.mean(all_squared_errors)
        print("Mean Squared Error (Classification):", mean_squared_error)
        return mean_squared_error

    def backProp_regression(self, results, label_index, data, initial_learning_rate=0.001, decay_rate=0.1, epoch=1, gradient_clip_value=0.5):
        learning_rate = initial_learning_rate * np.exp(-decay_rate * epoch)
        all_squared_errors = []
        
        num_layers = len(self.layers)

        for i, sample in enumerate(data):
            true_value = sample[label_index]  # Use actual value for regression
            
            output = results[i]
            # Calculate squared error
            squared_error = (output - true_value) ** 2  
            all_squared_errors.append(squared_error)

            # Calculate the derivative of the output error using the sigmoid derivative
            sample_derivative = (output - true_value) * sigmoid_derivative(output)

            # Backpropagation
            layer_errors = sample_derivative
            for layer_idx in range(num_layers - 1, -1, -1):
                layer = self.layers[layer_idx]
                new_errors = np.zeros((len(self.layers[layer_idx - 1]),)) if layer_idx > 0 else None

                for neuron_idx, neuron in enumerate(layer):
                    gradient = np.clip(layer_errors[neuron_idx], -gradient_clip_value, gradient_clip_value)
                    neuron.weights -= learning_rate * gradient * neuron.inputs
                    neuron.bias -= learning_rate * gradient

                    # Calculate errors for the previous layer if it exists
                    if layer_idx > 0:
                        for prev_neuron_idx in range(len(neuron.weights)):
                            new_errors[prev_neuron_idx] += gradient * neuron.weights[prev_neuron_idx]

                if layer_idx > 0:
                    layer_errors = new_errors

        mean_squared_error = np.mean(all_squared_errors)
        print("Mean Squared Error (Regression):", mean_squared_error/100)
        return mean_squared_error
