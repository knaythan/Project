import os
import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from PIL import ImageGrab, Image
import csv
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from model import CNNModel
import pickle

# Drawing prompt function
def draw_letter_prompt(letter_queue, save_folder, image_id, csv_writer):
    """
    A function to prompt the user to draw characters sequentially.

    Args:
        letter_queue (list): A list of letters to draw, processed sequentially.
        save_folder (str): Folder to save the images.
        image_id (int): Starting ID for saved images.
        csv_writer (csv.writer): CSV writer object to save image paths and labels.
    """
    os.makedirs(save_folder, exist_ok=True)

    # Initialize tkinter window
    root = tk.Tk()
    root.attributes("-fullscreen", True)

    canvas = tk.Canvas(root, width=1200, height=900, bg="white")
    canvas.pack()

    current_letter_index = [0]  # Use a mutable container to track the current index

    # Label to show the current letter
    letter_label = tk.Label(root, text=f"Draw the character: {letter_queue[0]}", font=("Helvetica", 24))
    letter_label.pack()

    def paint(event):
        x1, y1 = (event.x - 30), (event.y - 30)
        x2, y2 = (event.x + 30), (event.y + 30)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=1)
        

    def save():
        nonlocal image_id
        # Update canvas to render everything
        canvas.update()

        # Get the canvas's exact bounding box relative to the screen
        x = canvas.winfo_rootx() + 175
        y = canvas.winfo_rooty()
        x1 = x + canvas.winfo_width() + 175
        y1 = y + canvas.winfo_height() + 125

        # Capture the canvas area
        image = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")

        # Crop a few pixels from the left, right, top, and bottom to eliminate any unwanted background
        border_crop = 5  # Adjust this value to crop additional margins
        width, height = image.size
        cropped_image = image.crop((border_crop, border_crop, width - border_crop, height - border_crop))

        # Resize and save the image
        resized_image = cropped_image.resize((160, 120))  # Resize to the desired dimensions
        image_path = os.path.join(save_folder, f"{image_id:04d}.png")
        resized_image.save(image_path)
        print(f"Saved image to {image_path}")
        
        # Save the image path and label to the CSV file
        csv_writer.writerow([f"{image_id:04d}.png", letter_queue[current_letter_index[0]]])
        
        image_id += 1

        # Clear the canvas for the next letter
        clear()

        # Advance to the next letter
        current_letter_index[0] += 1
        if current_letter_index[0] < len(letter_queue):
            next_letter = letter_queue[current_letter_index[0]]
            letter_label.config(text=f"Draw the Letter: {next_letter}")
        else:
            print("All letters have been drawn. Closing the window.")
            root.destroy()

    def clear():
        canvas.delete("all")

    def exit_program():
        print("Exiting the program.")
        exit()
        root.destroy()

    # Bind events and add buttons
    canvas.bind("<B1-Motion>", paint)

    button_frame = tk.Frame(root)
    button_frame.pack()

    save_button = tk.Button(button_frame, text="Save", command=save)
    save_button.pack(side="left", padx=5)

    clear_button = tk.Button(button_frame, text="Clear", command=clear)
    clear_button.pack(side="left", padx=5)

    exit_button = tk.Button(button_frame, text="Exit", command=exit_program)
    exit_button.pack(side="left", padx=5)

    # Run the tkinter main loop
    root.mainloop()


# Custom Dataset for Reinforcement
class UserDrawnDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def reinforcement_loop(model, characters, user_images, save_folder, weights_folder, label_to_letter_dict_path, num_epochs=5, batch_size=16):
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    transform = ToTensor()

    # Load the label to letter dictionary from the pkl file
    with open(label_to_letter_dict_path, 'rb') as f:
        label_to_letter_dict = pickle.load(f)
        
    print(label_to_letter_dict)

    draw_counts = {char: 0 for char in characters}

    for char in characters:
        correct_predictions_in_a_row = 0
        while correct_predictions_in_a_row < 3:
            y_true, y_pred = [], []
            images, labels = [], []

            # Only focus on the current character
            for img in user_images[char]:
                images.append(img.astype(np.float32))  # Convert to float32
                labels.append(characters.index(char))

            dataset = UserDrawnDataset(images, labels, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training phase
            model.train()
            for epoch in range(num_epochs):
                for inputs, targets in loader:
                    # Ensure inputs have the correct shape [batch_size, 1, 120, 160]
                    if inputs.ndim == 3:  # If inputs are [batch_size, 120, 160]
                        inputs = inputs.unsqueeze(1)  # Add channel dimension

                    inputs = inputs.to(device).float()  # Cast to float32
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                for inputs, targets in loader:
                    # Ensure inputs have the correct shape [batch_size, 1, 120, 160]
                    if inputs.ndim == 3:  # If inputs are [batch_size, 120, 160]
                        inputs = inputs.unsqueeze(1)  # Add channel dimension

                    inputs = inputs.to(device).float()  # Cast to float32
                    outputs = model(inputs)  # Forward pass
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            # Translate labels to characters
            y_true_chars = [label_to_letter_dict[label] for label in y_true]
            y_pred_chars = [label_to_letter_dict[label] for label in y_pred]

            # Print the numerical label that the model guessed
            print(f"True labels: {y_true}")
            print(f"Predicted labels: {y_pred}")

            # Check if the model predicted correctly
            if all(np.array(y_true_chars) == np.array(y_pred_chars)):
                correct_predictions_in_a_row += 1
                print(f"Model predicted '{char}' correctly {correct_predictions_in_a_row} times in a row.")
            else:
                correct_predictions_in_a_row = 0
                print(f"Model did not predict '{char}' correctly. Collecting more data.")
                new_image_id = sum(len(user_images[k]) for k in user_images) + 1
                with open(os.path.join(save_folder, "image_mapping.csv"), "a", newline="") as file:
                    csv_writer = csv.writer(file)
                    draw_letter_prompt([char], save_folder, new_image_id, csv_writer)
                image_path = os.path.join(save_folder, f"{new_image_id:04d}.png")
                image = Image.open(image_path).resize((160, 120)).convert('L')
                user_images[char].append(np.array(image) / 255.0)
                draw_counts[char] += 1

    print("All characters have been trained and evaluated!")
    print("Draw counts:", draw_counts)

    # Save the reinforced model weights
    reinforced_weights_path = os.path.join(weights_folder, 'reinforced_cnn_weights.pth')
    torch.save(model.state_dict(), reinforced_weights_path)
    print(f"Reinforced model weights saved to {reinforced_weights_path}")


# Main
if __name__ == "__main__":
    characters = "0123456789Oom"
    save_folder = "user_drawings"
    weights_folder = "weights"
    label_to_letter_dict_path = os.path.join(weights_folder, "label_to_letter_dict.pkl")
    user_images = defaultdict(list)

    existing_characters = set()
    if os.path.exists(os.path.join(save_folder, "image_mapping.csv")):
        with open(os.path.join(save_folder, "image_mapping.csv"), "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                path, char = row
                existing_characters.add(char)
                image = Image.open(os.path.join(save_folder, path)).resize((160, 120))
                user_images[char].append(np.array(image) / 255.0)

    with open(os.path.join(save_folder, "image_mapping.csv"), "a", newline="") as file:
        csv_writer = csv.writer(file)
        if not existing_characters:
            csv_writer.writerow(["Path", "Label"])
        for char in characters:
            if char not in existing_characters:
                image_id = sum(len(user_images[k]) for k in user_images) + 1
                draw_letter_prompt([char], save_folder, image_id, csv_writer)
                image_path = os.path.join(save_folder, f"{image_id:04d}.png")
                image = Image.open(image_path).resize((160, 120))
                user_images[char].append(np.array(image) / 255.0)

    # Instantiate the model and load the weights
    num_classes = 62  # 10 digits + 26 lowercase + 26 uppercase
    model = CNNModel(num_classes=num_classes)

    # Initialize the fully connected layers
    input_shape = (1, 160, 120)
    model.initialize_fc_layers(input_shape)

    weights_path = './weights/cnn_weights.pth'
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    print("Model loaded successfully!")
    reinforcement_loop(model, characters, user_images, save_folder, weights_folder, label_to_letter_dict_path)
