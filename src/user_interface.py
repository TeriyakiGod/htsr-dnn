import tkinter as tk
import random

class UserInterface:
    def __init__(self, root, training_data, canvas_size=290, grid_size=29, border_size=5):
        self.root = root
        self.root.title("User Interface for painting digits")

        self.canvas_size = canvas_size
        self.grid_size = grid_size
        self.square_size = canvas_size // grid_size
        self.border_size = border_size
        self.training_data = training_data

        frame_width = canvas_size + 2 * (border_size + 10) + 300
        frame_height = canvas_size + 2 * (border_size + 10)
        self.root.geometry(f"{frame_width}x{frame_height}")

        self.frame = tk.Frame(self.root, bd=self.border_size, relief=tk.RIDGE)
        self.frame.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.frame, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.coordinates_label = tk.Label(self.root, text="Coordinates: (0, 0)")
        self.coordinates_label.pack(side=tk.TOP, pady=10)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_matrix)
        self.clear_button.pack(side=tk.TOP, pady=10)

        self.random_digit_button = tk.Button(self.root, text="Get Random Digit", command=self.get_random_digit)
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

        # Increase frame height to accommodate the text_widget
        frame_width = self.canvas_size + 2 * (self.border_size + 10) + 250
        frame_height = self.canvas_size + 2 * (self.border_size + 10) + 150
        self.root.geometry(f"{frame_width}x{frame_height}+{x_coordinate}+{y_coordinate}")

    def paint(self, event):
        x, y = event.x, event.y
        square_x = x // self.square_size
        square_y = y // self.square_size

        self.canvas.create_rectangle(
            square_x * self.square_size, square_y * self.square_size,
            (square_x + 1) * self.square_size, (square_y + 1) * self.square_size,
            fill="black"
        )

        similarity_percentages = self.calculate_similarity_percentages()
        self.update_text_widget(similarity_percentages)

        self.coordinates_label.config(text=f"Coordinates: ({square_x}, {square_y})")

    def calculate_similarity_percentages(self):
        # Add your logic here to calculate similarity percentages for digits 0 to 9
        # For now, let's assume random values
        return {i: random.uniform(0, 100) for i in range(10)}

    def update_text_widget(self, similarity_percentages):
        self.text_widget.delete(1.0, tk.END)  # Clear previous content

        # Find the digit with the highest percentage
        max_digit = max(similarity_percentages, key=similarity_percentages.get)

        # Sort the digits in ascending order
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

    def paint_square(self, row, col):
        self.canvas.create_rectangle(
            col * self.square_size, row * self.square_size,
            (col + 1) * self.square_size, (row + 1) * self.square_size,
            fill="black"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = UserInterface(root)
    root.mainloop()
