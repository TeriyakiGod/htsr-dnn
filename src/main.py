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

nn = NeuralNetwork(784, 89, 10)
learning_rate = 0.2
batch_size = 32
numberOfSteps = 1000
# Create batches
batches = []

for i in range(0, len(training_data), batch_size):
    batches.append(training_data[i:i + batch_size])
print("Start")

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.axis([0, numberOfSteps, 0, 1])
ax2.axis([0, len(batches), 0, 1])
ax1.clear()
for i in range(numberOfSteps):
    # Train on each batch
    ax2.clear()
    
    for j in range(len(batches)):
        nn.learn(batches[j], learning_rate)
        print("Batch: ", j, " Cost: ", nn.total_cost(batches[j]))
        ax2.scatter(j, nn.total_cost(batches[j]), marker='x')
        plt.pause(0.05)
    print("Step: ", i, " Cost: ", nn.total_cost(training_data))
    ax1.scatter(i, nn.total_cost(training_data), marker='x')
plt.show()
