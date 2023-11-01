##@package neural_network
# Neural network implementation.

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    ##Neural network class.

    def __init__(self, *layer_sizes):
        ##Neural network constructor.
        # @param layer_sizes (int[]): Number of nodes in each layer.

        self.layers = []  # Layer[len(layer_sizes) - 1]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def calculate_outputs(self, inputs):
        ##Calculate outputs of neural network.
        # @param inputs (float[]): Inputs to neural network.
        # @return float[]: Outputs of neural network.

        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs):
        ##Classify inputs.
        # @param inputs (float[]): Inputs to classify.
        # @return int: Class of inputs.

        outputs = self.calculate_outputs(inputs)
        return outputs.argmax()

    def visualize(self, graph_x, graph_y):
        ##Visualize neural network.
        # @param graph_x (float): X-axis value to graph.
        # @param graph_y (float): Y-axis value to graph.

        predicted_class = self.classify([graph_x, graph_y])

        if predicted_class == 0:
            plt.scatter(graph_x, graph_y, color="blue", label="Safe")
        elif predicted_class == 1:
            plt.scatter(graph_x, graph_y, color="red", label="Poisonous")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.show()


class Layer:
    ##Layer class.

    def __init__(self, numNodesIn, numNodesOut):
        ##Layer constructor.

        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut
        self.weights = np.zeros((self.numNodesIn, self.numNodesOut))
        self.biases = np.zeros(self.numNodesOut)

    def calculate_outputs(self, inputs):
        ##Calculate outputs of layer.
        # @param inputs (float[]): Inputs to layer.
        # @return float[]: Outputs of layer.

        weighted_inputs = np.zeros(self.numNodesOut)

        for nodeOut in range(self.numNodesOut):
            weighted_input = self.biases[nodeOut]
            for nodeIn in range(self.numNodesIn):
                weighted_input += inputs[nodeIn] * self.weights[nodeIn, nodeOut]
            weighted_inputs[nodeOut] = weighted_input

        return weighted_inputs
