import numpy as np
from data_loader import MnistDataloader
from neural_network import Layer, NeuralNetwork


import random
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join

def main():
    ## Create neural network:

    numberOfInputs = 784
    numberOfHiddenLayers = 2
    numberOfOutputs = 10
    neural_network = NeuralNetwork(numberOfInputs, numberOfHiddenLayers, numberOfOutputs)

    ## Create layer for testing:

    print("layer test:")
    numberNodesIn = 3
    numberNodesOut = 2
    layer = Layer(numberNodesIn, numberNodesOut)
    sample_inputs = np.random.rand(numberNodesIn)
    layer.print_outputs(sample_inputs) 

    ## Data load:

    images_filepath = "input/train-images-idx3-ubyte/train-images-idx3-ubyte"
    labels_filepath = "input/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    test_images_filepath = "input/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    test_labels_filepath = "input/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

    mnistLoader = MnistDataloader(images_filepath, labels_filepath, test_images_filepath, test_labels_filepath)

if __name__ == "__main__":
    main()