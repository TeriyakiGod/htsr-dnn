from data import DataPoint, MnistDataloader
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt


# Create a simple training dataset
mnist = MnistDataloader(
    "./input/train-images-idx3-ubyte/train-images-idx3-ubyte",
    "./input/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
    "./input/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    "./input/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
)
mnist_data, mnist_test = mnist.load_data()
images, labels = mnist_data

training_data = []
for i in range(2):
    flattened_image = [pixel for sublist in images[i] for pixel in sublist]
    training_data.append(DataPoint(flattened_image, labels[i], 10))

nn = NeuralNetwork(784, 89, 10)
learning_rate = 0.85
numberOfSteps = 1000
print("Start")
for i in range(numberOfSteps):
    nn.learn(training_data, learning_rate)
    print("Step: ", i, " Cost: ", nn.total_cost(training_data))