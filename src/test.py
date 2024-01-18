from data import DataPoint, MnistDataloader
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import os

nn = NeuralNetwork.load_model('model.pkl')

print("layers: ", len(nn.layers))
print("nodes in layer 1: ", (nn.layers[0].num_nodes_in))
print("nodes out layer 1: ", (nn.layers[0].num_nodes_out))
print("nodes in layer 2: ", (nn.layers[1].num_nodes_in))
print("nodes out layer 2: ", (nn.layers[1].num_nodes_out))

print("weights layer 1: ", len(nn.layers[0].weights))
print("weights layer 2: ", len(nn.layers[1].weights))

print("biases layer 1: ", len(nn.layers[0].biases))
print("biases layer 2: ", len(nn.layers[1].biases))