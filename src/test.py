from data import DataPoint, MnistDataloader
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import os
import numpy as np

class DataPoint:
    ##Data point.
    def __init__(self, inputs, label, numLabels):
        ##Data point constructor.
        # @param inputs (float[]): Inputs to the neural network.
        # @param label (int): Label of the data point.
        # @param numLabels (int): Number of labels in the dataset.
        self.inputs = inputs
        self.label = label
        self.expected_outputs = self.create_one_hot(label, numLabels)

    def create_one_hot(self, index, numLabels):
        one_hot = [0] * numLabels
        one_hot[index] = 1
        return one_hot

mnist = MnistDataloader(
    "./input/train-images-idx3-ubyte/train-images-idx3-ubyte",
    "./input/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
    "./input/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    "./input/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
)
mnist_data, mnist_test = mnist.load_data()
images, labels = mnist_test

test_data = []
for j in range(1000):
    flattened_image = [pixel for sublist in images[j] for pixel in sublist]
    test_data.append(DataPoint(flattened_image, labels[j], 10))

nn = NeuralNetwork(784, 89, 10)
if os.path.exists('model.pkl'):
    nn = NeuralNetwork.load_model('model.pkl')

# Initialize an empty list to store the network's predictions
predictions = []

# Iterate over the test samples
for sample in test_data:
    # Forward pass through the network to obtain the predicted output
    output = nn.calculate_outputs(sample.inputs)
    
    # Store the predicted output in the list of predictions
    predictions.append(np.argmax(output))
    

# Initialize a counter to keep track of the number of correct predictions
correct = 0
# Evaluate the accuracy of the network
for i in range(len(predictions)):
    if predictions[i] == test_data[i].label:
        correct += 1

# Calculate the accuracy as a percentage
accuracy = (correct / len(predictions)) * 100

print(f"Accuracy: {accuracy:.2f}%")