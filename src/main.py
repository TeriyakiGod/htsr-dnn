from data import DataPoint
from neural_network import NeuralNetwork

# Create a simple training dataset
training_data = [
    # AND function: y = x1 AND x2
    [
        DataPoint([0, 0], 0, 2),
        DataPoint([0, 1], 0, 2),
        DataPoint([1, 0], 0, 2),
        DataPoint([1, 1], 1, 2),
    ],
    # OR function: y = x1 OR x2
    [
        DataPoint([0, 0], 0, 2),
        DataPoint([0, 1], 1, 2),
        DataPoint([1, 0], 1, 2),
        DataPoint([1, 1], 1, 2),
    ],
    # XOR function: y = x1 XOR x2
    [
        DataPoint([0, 0], 0, 2),
        DataPoint([0, 1], 1, 2),
        DataPoint([1, 0], 1, 2),
        DataPoint([1, 1], 0, 2),
    ],
]

# Test the neural network
for test in training_data:
    nn = NeuralNetwork(2, 10, 2)
    learning_rate = 0.2

    # Train the neural network
    for _ in range(2000):
        nn.learn(test, learning_rate)
    print("Input | Expected Output | Actual Output")
    print("------|-----------------|--------------")
    for data_point in test:
        inputs = data_point.inputs
        expected_output = data_point.expected_outputs
        actual_output = nn.calculate_outputs(inputs)
        actual_output[0] = 1 if actual_output[0] > 0.5 else 0
        actual_output[1] = 1 if actual_output[1] > 0.5 else 0
        print(f"{inputs}      {expected_output}            {actual_output}")
