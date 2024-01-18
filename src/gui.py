from data import DataPoint, MnistDataloader
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import os
import time
import user_interface as ui
import tkinter as tk

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

paint_root = tk.Tk()
paint_app = ui.UserInterface(root=paint_root, training_data=test_data, neural_network=nn)
paint_root.mainloop()