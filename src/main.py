from data import DataPoint
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt


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
fig, axs = plt.subplots(len(training_data), 1, figsize=(5, 15))  # Create subplots for each test

for i, test in enumerate(training_data):
    nn = NeuralNetwork(2, 10, 2)
    learning_rate = 0.3
    numberOfSteps = 1000
    accuracies = []  # List to store accuracies

    # Train the neural network
    for _ in range(numberOfSteps):
        nn.learn(test, learning_rate)
        
        # Calculate accuracy
        correct_predictions = 0
        for data_point in test:
            inputs = data_point.inputs
            expected_output = data_point.expected_outputs
            actual_output = nn.calculate_outputs(inputs)
            if (actual_output[0] > 0.5) == expected_output[0] and (actual_output[1] > 0.5) == expected_output[1]:
                correct_predictions += 1
        accuracy = correct_predictions / len(test)
        accuracies.append(accuracy)

    # Plot accuracy over iterations
    axs[i].plot(accuracies)
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()