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
        total_cost = 0
        for data_point in data:
            total_cost += self.cost(data_point)
        return total_cost / len(data)

    def learn(self, training_data, learning_rate):
        h = 0.0001
        original_cost = self.total_cost(training_data)
        for layer in self.layers:
            for node_in in range(layer.num_nodes_in):
                for node_out in range(layer.num_nodes_out):
                    layer.weights[node_in, node_out] += h
                    delta_cost = self.total_cost(training_data) - original_cost
                    layer.weights[node_in, node_out] -= h
                    layer.cost_gradient_w[node_in, node_out] = delta_cost / h
            for bias_index in range(len(layer.biases)):
                layer.biases[bias_index] += h
                delta_cost = self.total_cost(training_data) - original_cost
                layer.biases[bias_index] -= h
                layer.cost_gradient_b[bias_index] = delta_cost / h
        self.apply_gradients(learning_rate)

    def apply_gradients(self, learning_rate):
        for layer in self.layers:
            layer.apply_gradients(learning_rate)


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

    def back_propagation(self, output_activation, expected_output):
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
