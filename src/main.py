from data import DataPoint, MnistDataloader
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import time
import user_interface as ui
import tkinter as tk
import os

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
for i in range(1000):
    flattened_image = [pixel for sublist in images[i] for pixel in sublist]
    training_data.append(DataPoint(flattened_image, labels[i], 10))

nn = NeuralNetwork(784, 2, 10)
learning_rate = 0.9
batch_size = 128
numberOfSteps = 1
batches = []
nn = NeuralNetwork(784, 89, 10)
# Load the last saved model if it exists
if os.path.exists('model.pkl'):
    nn = NeuralNetwork.load_model('model.pkl')

learning_rate = 0.5
batch_size = 1024
numberOfSteps = 1000

# Create a figure for plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

training_data = []
for j in range(len(images)):
    flattened_image = [pixel for sublist in images[j] for pixel in sublist]
    training_data.append(DataPoint(flattened_image, labels[j], 10))
batches = []
for j in range(0, len(training_data), batch_size):
    batches.append(training_data[j:j + batch_size])

print("Data loaded. Starting training...")
start_time = time.time()

for i in range(numberOfSteps):
    step_start_time = time.time()

    for j in range(len(batches)):
        nn.learn(batches[j], learning_rate)

    step_end_time = time.time()
    step_duration = step_end_time - step_start_time

    print(f"Step: {i}, Cost: {nn.total_cost(training_data)}, Time: {step_duration} seconds")

end_time = time.time()
total_duration = end_time - start_time
print(f"Training complete. Total time: {total_duration} seconds")

paint_root = tk.Tk()
paint_app = ui.UserInterface(root=paint_root, training_data=training_data)
paint_root.mainloop()

for i in range(numberOfSteps):
    ax2.clear()
    for j in range(len(batches)):
        nn.learn(batches[j], learning_rate)
        print("Step: ", i, "Batch: ", j, " Cost: ", nn.total_cost(batches[j]))
        ax2.scatter(j, nn.total_cost(batches[j]), marker='x')
        plt.pause(0.01)
    nn.save_model('model.pkl')
    print("Step: ", i, " Cost: ", nn.total_cost(training_data))
    learning_rate *= 0.95
    ax1.scatter(i, nn.total_cost(training_data), marker='x')
plt.show()