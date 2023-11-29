from data import DataPoint
from neural_network import NeuralNetwork

# Create a simple training dataset
# XOR function: y = x1 XOR x2
training_data = [
    DataPoint([0, 0], 0, 2),
    DataPoint([0, 1], 1, 2),
    DataPoint([1, 0], 1, 2),
    DataPoint([1, 1], 0, 2),
]

# Create a neural network with 2 hidden nodes, and 1 output
nn = NeuralNetwork(2, 2)

# Train the neural network
for _ in range(2):
    nn.learn(training_data, 0.1)

# Test the neural network
print("XOR Test")
print(
    "Inputs [true, false]\t\tExpected Output [true, false]\t\tActual Output [true, false]"
)
print(
    "---------------------------------------------------------------------------------------------------------"
)
for data_point in training_data:
    inputs = data_point.inputs
    expected_output = data_point.expected_outputs
    actual_output = nn.calculate_outputs(inputs)
    print(f"\t{inputs}\t\t\t\t{expected_output}\t\t\t{actual_output}")
