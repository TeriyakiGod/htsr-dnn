import numpy as np
from neural_network import Layer, NeuralNetwork


def main():
    neural_network = NeuralNetwork(784, 2, 10)

    layer = Layer(6, 3)
    sample_inputs = np.random.rand(6)
    layer.print_outputs(sample_inputs) 

if __name__ == "__main__":
    main()
