import numpy as np
from neural_network import Layer, NeuralNetwork


def main():
    numberOfInputs = 784
    numberOfHiddenLayers = 2
    numberOfOutputs = 10
    neural_network = NeuralNetwork(numberOfInputs, numberOfHiddenLayers, numberOfOutputs)

    numberNodesIn = 6
    numberNodesOut = 3
    layer = Layer(numberNodesIn, numberNodesOut)
    sample_inputs = np.random.rand(numberNodesIn)
    layer.print_outputs(sample_inputs) 

if __name__ == "__main__":
    main()
