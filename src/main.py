import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


def main():
    neural_network = NeuralNetwork(2, 3, 2)
    neural_network.visualize(5.0, 2.0)


if __name__ == "__main__":
    main()
