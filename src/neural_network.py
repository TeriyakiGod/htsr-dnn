"""
Module description: Briefly explain the purpose and contents of the module.
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Description of NeuralNetwork.

    Attributes:
    layers (Layer[]): Array of type Layer.

    Methods:
    method1(self, arg1, arg2) -> return_type: Description of method 1.

    """

    def __init__(self, *layer_sizes):
        self.layers = [] #Layer[len(layer_sizes) - 1]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def calculate_outputs(self, inputs):
        """
        Calculates the outputs of the neural network based on the given inputs.

        Args:
        inputs: Array containing input values.

        Returns:
        float: Array containing calculated outputs.
        """
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    def classify(self, inputs):
        """
        Run the inputs throught the network and 
        calculate which output node has the highest value

        Args:
        inputs: Array containing input values.

        Returns:
        float: Max value of calculated outputs
        """
        outputs = self.calculate_outputs(inputs)
        return outputs.argmax()

    def visualize(self, graph_x, graph_y):
        predicted_class = self.classify([graph_x, graph_y])

        if predicted_class == 0:
            plt.scatter(graph_x, graph_y, color='blue', label='Safe')
        elif predicted_class == 1:
            plt.scatter(graph_x, graph_y, color='red', label='Poisonous')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()

class Layer:
    """
    Represents a single layer in neural network.

    Attributes:
    numNodesIn : Number of nodes that comes in.
    numNodesOut: Number of nodes that comes out.

    Methods:
    calculate_outputs(self, inputs) -> returns weighted inputs
    """
    def __init__(self, numNodesIn, numNodesOut):
        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut
        self.weights = np.zeros((self.numNodesIn, self.numNodesOut))
        self.biases = np.zeros(self.numNodesOut)

    def calculate_outputs(self, inputs):
        """
        Calculates the output of a layer based on the inputs, weights, and biases.

        Args:
        inputs: Array containing input values. In our scenario these are just each pixel.

        Returns:
        float: Array of floats containing calculated outputs of an layer.
        """
        weighted_inputs = np.zeros(self.numNodesOut)

        for nodeOut in range(self.numNodesOut):
            weighted_input = self.biases[nodeOut]
            for nodeIn in range(self.numNodesIn):
                weighted_input += inputs[nodeIn] * self.weights[nodeIn, nodeOut]
            weighted_inputs[nodeOut] = weighted_input
        
        return weighted_inputs