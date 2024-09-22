import tkinter as tk
import random

class RandomValueGeneratorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Random Value Generator")

        # List of values
        self.values = ["Mario Kart", "Mario Bros", "Pokemon"]

        # Variable to hold the random value
        self.random_value_var = tk.StringVar()

        # Label to display the random value
        self.value_label = tk.Label(master, textvariable=self.random_value_var, font=("Helvetica", 16))
        self.value_label.pack(pady=10)

        # Button
        self.button = tk.Button(master, text="Generate Random Value", command=self.generate_random_value)
        self.button.pack(pady=10)

    def generate_random_value(self):
        # Get a random value from the list
        random_value = random.choice(self.values)

        # Update the random value in the interface
        self.random_value_var.set(random_value)

# Create the main application window
root = tk.Tk()

# Create an instance of the RandomValueGeneratorApp
app = RandomValueGeneratorApp(root)

# Run the Tkinter event loop
root.mainloop()
