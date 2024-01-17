import numpy as np
import activation_functions as af
import pickle
from numba import cuda, jit


class NeuralNetwork:
    def __init__(self, *nOfNodesInLayers):
        self.layers = [Layer(nOfNodesInLayers[i], nOfNodesInLayers[i + 1]) for i in range(len(nOfNodesInLayers) - 1)]

    @jit
    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    @jit
    def cost(self, data_point):
        outputs = self.calculate_outputs(data_point.inputs)
        output_layer = self.layers[-1]
        cost = 0

        for i in range(len(outputs)):
            cost += output_layer.node_cost(outputs[i], data_point.expected_outputs[i])
        return cost

    @jit
    def total_cost(self, data):
        total_cost = sum(self.cost(data_point) for data_point in data)
        return total_cost / len(data)

    @jit
    def learn(self, training_data, learning_rate):
        for data_point in training_data:
            self.backpropagate(data_point)
        self.apply_gradients(learning_rate)

    @jit
    def backpropagate(self, data_point):
        outputs = self.calculate_outputs(data_point.inputs)
        deltas = [output - expected_output for output, expected_output in zip(outputs, data_point.expected_outputs)]

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            inputs = data_point.inputs if i == 0 else self.layers[i - 1].calculate_outputs(data_point.inputs)
            layer.calculate_gradients(inputs, deltas)
            if i > 0 and i < len(self.layers) - 1:
                deltas = np.dot(layer.weights, deltas) * af.sigmoid_derivative(layer.calculate_outputs(data_point.inputs))

    @jit
    def apply_gradients(self, learning_rate):
        for layer in self.layers:
            if self.layers.index(layer) == 0:
                continue
            else:
                layer.apply_gradients(learning_rate)

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model

class Layer:
    def __init__(self, num_nodes_in, num_nodes_out):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        # Xavier/Glorot initialization for weights
        self.weights = np.random.randn(self.num_nodes_in, self.num_nodes_out) * np.sqrt(2.0 / (self.num_nodes_in + self.num_nodes_out))
        self.biases = np.random.randn(self.num_nodes_out)
        self.cost_gradient_w = np.zeros((self.num_nodes_in, self.num_nodes_out))
        self.cost_gradient_b = np.zeros(self.num_nodes_out)
        self.weight_constant = 0.9

    @jit
    def node_cost(self, output_activation, expected_output):
        error = expected_output - output_activation
        return error * error

    @jit
    def calculate_outputs(self, inputs):
        weighted_inputs = np.dot(inputs, self.weights) + self.biases
        outputs = af.sigmoid(weighted_inputs)
        return outputs

    @jit
    def calculate_gradients(self, inputs, deltas):
        self.cost_gradient_w = np.outer(inputs, deltas)
        self.cost_gradient_b = np.array(deltas)

    @jit
    def apply_gradients(self, learning_rate):
        self.weights -= learning_rate * self.cost_gradient_w
        self.biases -= learning_rate * self.cost_gradient_b
