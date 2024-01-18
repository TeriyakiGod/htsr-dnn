import tkinter as tk
import user_interface as ui
from neural_network import NeuralNetwork
from data import DataPoint, MnistDataloader

nn = NeuralNetwork.load_model('model.pkl')

mnist = MnistDataloader(
    "./input/train-images-idx3-ubyte/train-images-idx3-ubyte",
    "./input/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
    "./input/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    "./input/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
)

mnist_data, mnist_test = mnist.load_data()
images, labels = mnist_data

training_data = []
for j in range(1000):
    flattened_image = [pixel for sublist in images[j] for pixel in sublist]
    training_data.append(DataPoint(flattened_image, labels[j], 10))

mnist_data, mnist_test = mnist.load_data()
images, labels = mnist_data
training_data = []


paint_root = tk.Tk()
paint_app = ui.UserInterface(root=paint_root, training_data=training_data, neural_network=nn)
paint_root.mainloop()