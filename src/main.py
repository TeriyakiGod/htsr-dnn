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
for i in range(5000):
    flattened_image = [pixel for sublist in images[i] for pixel in sublist]
    training_data.append(DataPoint(flattened_image, labels[i], 10))

nn = NeuralNetwork(784, 2, 10)
learning_rate = 0.85
batch_size = 128
numberOfSteps = 50
batches = []

for i in range(0, len(training_data), batch_size):
    batches.append(training_data[i:i + batch_size])

print("Data loaded. Starting training...")
for i in range(numberOfSteps):
    for j in range(len(batches)):
        nn.learn(batches[j], learning_rate)
    print("Step: ", i, " Cost: ", nn.total_cost(training_data))