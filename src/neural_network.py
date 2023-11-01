##@package neural_network
# Neural network implementation.

# Take all values from connected neurons multiplied by their respective weight, 
# add them, and apply an activation function. 
# Then, the neuron is ready to send its new value to other neurons.

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    ##Neural Network.

    def __init__(self, numOfInputs, numOfHiddenLayers, numOfOutputs):
        ##Neural network constructor.
        # @param numOfInputs (int): Number of input nodes.
        # @param numOfHiddenLayers (int): Number of hidden layers.
        # @param numOfOutputs (int): Number of output nodes.
        self.numOfInputs = numOfInputs
        self.numOfHiddenLayers = numOfHiddenLayers
        self.numOfOutputs = numOfOutputs
        self.layers = []
        self.create_layers(numOfInputs, numOfHiddenLayers, numOfOutputs)
        self.print_info_about_architecture()

    def create_layers(self, numOfInputs, numOfHiddenLayers, numOfOutputs):
        ##Creates layers for the neural network.
        # @param numOfInputs (int): Number of input nodes.
        # @param numOfHiddenLayers (int): Number of hidden layers.
        # @param numOfOutputs (int): Number of output nodes.
        inputLayer = Layer(numOfInputs, np.random.randint(5, 15))
        self.layers.append(inputLayer)

        for i in range(numOfHiddenLayers):
            if i == 0:
                hiddenLayer = Layer(numOfInputs, np.random.randint(5, 15))
                self.layers.append(hiddenLayer)
            else:
                hiddenLayer = Layer(self.layers[-1].numNodesOut, np.random.randint(5, 15))
                self.layers.append(hiddenLayer)

        outputLayer = Layer(self.layers[-1].numNodesOut, numOfOutputs)
        self.layers.append(outputLayer)

    def print_info_about_architecture(self):
        ##Prints information about the neural network's architecture.
        print("Initializing Neural Network with:")
        for i, layer in enumerate(self.layers):
            if i == 0:
                print(f"Input Layer: {layer.numNodesIn} nodes")
            elif i == len(self.layers) - 1:
                print(f"Output Layer: {layer.numNodesOut} nodes and {layer.numNodesIn} inputs")
            else:
                print(f"Hidden Layer {i}: {layer.numNodesOut} nodes and {layer.numNodesIn} inputs")

class Layer:
    ##Layer within a Neural Network.

    def __init__(self, numNodesIn, numNodesOut):
        ##Layer constructor.

        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut
        self.weights = np.random.rand(self.numNodesIn, self.numNodesOut)
        self.biases = np.random.randn(self.numNodesOut)

    def calculate_outputs(self, inputs):
        ##Calculates outputs of the layer based on the given inputs.
        # @param inputs (float[]): Inputs to the layer.
        # @return float[]: Outputs of the layer.

        weighted_inputs = np.dot(inputs, self.weights) + self.biases # Iloczyn skalarny

        return weighted_inputs
    
    def print_outputs(self, inputs):
        ##Prints generated weights, biases and calculated outputs of a Layer.
        np.set_printoptions(precision=2)
        print("Weights:")
        print(self.weights)
        print("Biases:")
        print(self.biases)
        outputs = self.calculate_outputs(inputs)
        print("Outputs:")
        print(outputs)
