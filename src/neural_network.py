##@package neural_network
# Neural network implementation.

import numpy as np
import activation_functions as af


class NeuralNetwork:
    ##Neural Network.

    def __init__(self, *nOfNodesInLayers):
        ##Neural network constructor.
        # @param nOfLayers (int): Number of nodes in each layer: (hidden, hidden, ..., output)
        # The last argument is the number of output nodes.
        # The arguments in between are the number of nodes in each hidden layer.
        self.layers = []
        for i in range(len(nOfNodesInLayers) - 1):
            self.layers.append(Layer(nOfNodesInLayers[i], nOfNodesInLayers[i + 1]))

    def calculate_outputs(self, inputs):
        ##Calculates outputs of the neural network based on the given inputs.
        # @param inputs (float[]): Inputs to the neural network.
        # @return float[]: Outputs of the neural network.

        outputs = inputs
        for layer in self.layers:
            outputs = layer.calculate_outputs(outputs)
        return outputs

    def cost(self, data_point):
        outputs = self.calculate_outputs(data_point.inputs)
        output_layer = self.layers[-1]
        cost = 0

        for i in range(len(outputs)):
            cost += output_layer.node_cost(outputs[i], data_point.expected_outputs[i])
        return cost

    def total_cost(self, data):
        outputs = np.array([self.calculate_outputs(dp.inputs) for dp in data])
        expected_outputs = np.array([dp.expected_outputs for dp in data])
        return np.mean(np.sum((outputs - expected_outputs) ** 2, axis=-1))


    def learn(self, training_data, learning_rate, batch_size):
        h = 0.0001
        original_cost = self.total_cost(training_data)

        # Shuffle the training data to introduce randomness
        np.random.shuffle(training_data)

        for start in range(0, len(training_data), batch_size):
            end = start + batch_size
            batch = training_data[start:end]

            # Accumulate gradient updates over the batch
            for layer in self.layers:
                layer_gradient_w = np.zeros_like(layer.weights)
                layer_gradient_b = np.zeros_like(layer.biases)

                for data_point in batch:
                    # Calculate gradients for weights using vectorized operations
                    delta_cost_w = np.zeros_like(layer.weights)
                    for node_in in range(layer.num_nodes_in):
                        for node_out in range(layer.num_nodes_out):
                            layer.weights[node_in, node_out] += h
                            delta_cost_w[node_in, node_out] = self.total_cost(batch) - original_cost
                            layer.weights[node_in, node_out] -= h

                    # Calculate gradients for biases using vectorized operations
                    delta_cost_b = np.zeros_like(layer.biases)
                    for bias_index in range(len(layer.biases)):
                        layer.biases[bias_index] += h
                        delta_cost_b[bias_index] = self.total_cost(batch) - original_cost
                        layer.biases[bias_index] -= h

                    # Accumulate gradients over the batch
                    layer_gradient_w += delta_cost_w / h
                    layer_gradient_b += delta_cost_b / h

                # Update weights and biases using accumulated gradients
                layer.weights -= learning_rate * (layer_gradient_w / len(batch))
                layer.biases -= learning_rate * (layer_gradient_b / len(batch))

        self.apply_gradients(learning_rate)




class Layer:
    ##Layer within a Neural Network.

    def __init__(self, num_nodes_in, num_nodes_out):
        ##Layer constructor.
        # @param num_nodes_in (int): Number of nodes in the previous layer.
        # @param num_nodes_out (int): Number of nodes in the current layer.

        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.weights = np.random.rand(self.num_nodes_in, self.num_nodes_out)
        self.biases = np.random.randn(self.num_nodes_out)
        self.cost_gradient_w = np.zeros((self.num_nodes_in, self.num_nodes_out))
        self.cost_gradient_b = np.zeros(self.num_nodes_out)

    def node_cost(self, output_activation, expected_output):
        error = expected_output - output_activation
        return error * error

    def apply_gradients(self, learning_rate):
        ##Applies the gradients to the weights and biases.
        # @param learning_rate (float): Learning rate of the neural network.

        self.biases -= learning_rate * self.cost_gradient_b
        self.weights -= learning_rate * self.cost_gradient_w

    def calculate_outputs(self, inputs):
        ##Calculates outputs of the layer based on the given inputs.
        # @param inputs (float[]): Inputs to the layer.
        # @return float[]: Outputs of the layer.

        weighted_inputs = np.dot(inputs, self.weights) + self.biases
        outputs = []
        for weighted_input in weighted_inputs:
            outputs.append(af.sigmoid(weighted_input))
        return outputs
