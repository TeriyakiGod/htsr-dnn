import tkinter as tk
import random

import numpy as np

class UserInterface:
    def __init__(self, root, training_data, neural_network, canvas_size=290, grid_size=29, border_size=5):
        self.root = root
        self.root.title("User Interface for painting digits")

        self.canvas_size = canvas_size
        self.grid_size = grid_size
        self.square_size = canvas_size // grid_size
        self.border_size = border_size
        self.training_data = training_data
        self.neural_network = neural_network

        frame_width = canvas_size + 2 * (border_size + 10) + 300
        frame_height = canvas_size + 2 * (border_size + 10)
        self.root.geometry(f"{frame_width}x{frame_height}")

        self.frame = tk.Frame(self.root, bd=self.border_size, relief=tk.RIDGE)
        self.frame.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.frame, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<B2-Motion>", self.paint)

        self.coordinates_label = tk.Label(self.root, text="Coordinates: (0, 0)")
        self.coordinates_label.pack(side=tk.TOP, pady=10)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_matrix)
        self.clear_button.pack(side=tk.TOP, pady=10)

        self.random_digit_button = tk.Button(self.root, text="Random Digit from dataset", command=self.get_random_digit)
        self.random_digit_button.pack(side=tk.TOP, pady=10)

        self.expected_output_text = tk.Label(self.root, text="Expected output: ")
        self.expected_output_text.pack(side=tk.TOP, pady=10)

        self.text_widget = tk.Text(self.root, height=10, width=20)
        self.text_widget.pack(side=tk.RIGHT, padx=10)

        self.center_window()

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x_coordinate = (screen_width - self.canvas_size) // 2
        y_coordinate = (screen_height - self.canvas_size) // 2

        frame_width = self.canvas_size + 2 * (self.border_size + 10) + 250
        frame_height = self.canvas_size + 2 * (self.border_size + 10) + 150
        self.root.geometry(f"{frame_width}x{frame_height}+{x_coordinate}+{y_coordinate}")

    def paint(self, event):
        x, y = event.x, event.y
        square_x = x // self.square_size
        square_y = y // self.square_size

        if 0 <= square_x < self.grid_size and 0 <= square_y < self.grid_size:
            if event.state & 0x1:
                self.canvas.create_rectangle(
                    square_x * self.square_size, square_y * self.square_size,
                    (square_x + 1) * self.square_size, (square_y + 1) * self.square_size,
                    fill="red"
                )
            else:
                self.canvas.create_rectangle(
                    square_x * self.square_size, square_y * self.square_size,
                    (square_x + 1) * self.square_size, (square_y + 1) * self.square_size,
                    fill="black"
                )

            similarity_percentages = self.calculate_similarity_percentages(self.get_painted_matrix(), self.neural_network)
            self.update_text_widget(similarity_percentages)
            self.coordinates_label.config(text=f"Coordinates: ({square_x}, {square_y})")


    def calculate_similarity_percentages(self, painted_matrix, neural_network):
        inputs = [255 if value > 0 else 0 for value in painted_matrix]
        inputs = inputs[:784]
        outputs = neural_network.calculate_outputs(inputs)

        similarity_percentages = {i: output * 100 for i, output in enumerate(outputs)}
        return similarity_percentages
        #return {i: random.uniform(0, 100) for i in range(10)}

    def update_text_widget(self, similarity_percentages):
        self.text_widget.delete(1.0, tk.END)
        max_digit = max(similarity_percentages, key=similarity_percentages.get)
        sorted_digits = sorted(similarity_percentages.keys())

        for digit in sorted_digits:
            percentage = similarity_percentages[digit]
            background_color = "lightgreen" if digit == max_digit else "white"
            digit_text = f"Digit {digit}"
            self.text_widget.insert(tk.END, f"{digit_text}: {percentage:.2f}%\n")
            self.text_widget.tag_configure(f"bg_{digit + 1}", background=background_color)
            self.text_widget.tag_add(f"bg_{digit + 1}", f"{digit + 1}.0", f"{digit + 2}.0")

    def clear_matrix(self):
        self.expected_output_text.config(text="Expected output: ")
        painted_squares = self.canvas.find_all()
        for square in painted_squares:
            self.canvas.delete(square)

        self.text_widget.delete(1.0, tk.END)
        self.coordinates_label.config(text="Coordinates: (0, 0)")

    def get_random_digit(self):
        self.clear_matrix()
        if self.training_data:
            random_data_point = random.choice(self.training_data)
            for i, value in enumerate(random_data_point.inputs):
                if value > 0:
                    row, col = divmod(i, 28)
                    self.paint_square(row, col)

            self.expected_output_text.config(text=f"Expected output: {random_data_point.label}")

    def get_painted_matrix(self):
        matrix = np.zeros((self.grid_size, self.grid_size))
        painted_squares = self.canvas.find_all()

        for square in painted_squares:
            coordinates = self.canvas.coords(square)
            square_x = int(coordinates[0] // self.square_size)
            square_y = int(coordinates[1] // self.square_size)
            matrix[square_y, square_x] = 1

        return matrix.flatten()

    def paint_square(self, row, col):
        self.canvas.create_rectangle(
            col * self.square_size, row * self.square_size,
            (col + 1) * self.square_size, (row + 1) * self.square_size,
            fill="black"
        )

    def calculate_outputs_nn(self, neural_network, painted_matrix):
        inputs = [255 if value > 0 else 0 for value in painted_matrix]
        outputs = neural_network.calculate_outputs(inputs)

        for i, percentage in enumerate(outputs):
            background_color = "lightgreen" if i == np.argmax(outputs) else "white"
            digit_text = f"Digit {i}"
            self.text_widget.insert(tk.END, f"{digit_text}: {percentage * 100:.2f}%\n")
            self.text_widget.tag_configure(f"bg_{i + 1}", background=background_color)
            self.text_widget.tag_add(f"bg_{i + 1}", f"{i + 1}.0", f"{i + 2}.0")

        self.expected_output_text.config(text=f"Expected output: ")
        # return outputs ...

if __name__ == "__main__":
    root = tk.Tk()
    app = UserInterface(root)
    root.mainloop()
