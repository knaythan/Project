import os
import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from PIL import ImageGrab, Image
import csv
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from model import CNNModel
import pickle
import sys
import random
import argparse

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
    root.state('zoomed')  # Maximize the window

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
        y1 = y + canvas.winfo_height() + 150

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

    # Bind the window close event to the exit_program function
    root.protocol("WM_DELETE_WINDOW", exit_program)

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
    
def reinforcement_loop(model, label_to_letter_dict, user_images, save_folder, weights_folder, num_epochs=5, batch_size=16, prefix=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    transform = ToTensor()

    # Reverse the label-to-letter dictionary to get letter-to-label mapping
    letter_to_label_dict = {v: k for k, v in label_to_letter_dict.items()}

    print("Label to letter mapping:", label_to_letter_dict)

    draw_counts = {char: 0 for char in label_to_letter_dict.values()}

    trained_characters = []

    def calculate_global_accuracy():
        """Randomly select one image per trained character and evaluate global accuracy."""
        correct_predictions = 0
        for char in trained_characters:
            if char not in user_images or len(user_images[char]) == 0:
                continue

            # Randomly pick one image for this character
            image = random.choice(user_images[char])
            label = letter_to_label_dict[char]

            # Preprocess and predict
            model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                output = model(input_tensor)
                _, pred = torch.max(output, 1)

            if pred.item() == label:
                correct_predictions += 1

        if len(trained_characters) == 0:
            return 0.0

        return correct_predictions / len(trained_characters)

    for char in label_to_letter_dict.values():
        correct_prediction = False
        retries = 0

        while not correct_prediction and retries < 3:
            y_true, y_pred = [], []
            images, labels = [], []

            # Ensure each character has at least 5 images
            while len(user_images[char]) < 5:
                new_image_id = sum(len(user_images[k]) for k in user_images) + 1
                with open(os.path.join(save_folder, "image_mapping.csv"), "a", newline="") as file:
                    csv_writer = csv.writer(file)
                    draw_letter_prompt([char], save_folder, new_image_id, csv_writer)
                image_path = os.path.join(save_folder, f"{new_image_id:04d}.png")
                image = Image.open(image_path).resize((160, 120)).convert('L')
                user_images[char].append(np.array(image) / 255.0)
                draw_counts[char] += 1

            # Prepare data for training
            for other_char in label_to_letter_dict.values():
                sampled_images = user_images.get(other_char, [])
                labels.extend([letter_to_label_dict[other_char]] * len(sampled_images))
                images.extend(sampled_images)

            # Shuffle data
            combined = list(zip(images, labels))
            random.shuffle(combined)
            images, labels = zip(*combined)

            # Create the dataset
            dataset = UserDrawnDataset(list(images), list(labels), transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training phase
            model.train()
            total_correct = 0
            total_samples = 0
            for epoch in range(num_epochs):
                for inputs, targets in loader:
                    if inputs.ndim == 3:
                        inputs = inputs.unsqueeze(1)

                    inputs = inputs.to(device).float()
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == targets).sum().item()
                    total_samples += targets.size(0)

            training_accuracy = total_correct / total_samples
            print(f"Training accuracy for '{char}': {training_accuracy * 100:.2f}%")

            # Evaluation phase
            model.eval()
            batch_correct = 0
            total_batch_samples = 0

            with torch.no_grad():
                for inputs, targets in loader:
                    if inputs.ndim == 3:
                        inputs = inputs.unsqueeze(1)

                    inputs = inputs.to(device).float()
                    targets = targets.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    batch_correct += (preds == targets).sum().item()
                    total_batch_samples += targets.size(0)

            batch_accuracy = batch_correct / total_batch_samples
            global_accuracy = calculate_global_accuracy()

            print(f"Batch accuracy for '{char}': {batch_accuracy * 100:.2f}%")
            print(f"Global accuracy across trained and tested characters: {global_accuracy * 100:.2f}%")

            if batch_accuracy >= 0.85 and global_accuracy >= 0.85:
                correct_prediction = True
                trained_characters.append(char)
                print(f"Model predicted '{char}' correctly with sufficient accuracy.")
            else:
                print(f"Retrying with existing data for '{char}'.")
                retries += 1

        if not correct_prediction:
            print(f"Exhausted retries for '{char}'. Collecting more data.")
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

    if prefix != "":
        reinforced_weights_path = os.path.join(weights_folder, f"{prefix}_reinforced_cnn_weights.pth")
    else:
        reinforced_weights_path = os.path.join(weights_folder, "reinforced_cnn_weights.pth")
    torch.save(model.state_dict(), reinforced_weights_path)
    print(f"Reinforced model weights saved to {reinforced_weights_path}")

# Main
if __name__ == "__main__":
    save_folder = "user_drawings"
    user_images = defaultdict(list)
    weights_folder = "weights"

    parser = argparse.ArgumentParser(description="Evaluate Character Classification Model")
    parser.add_argument("--test", type=str, nargs='?', const='lower', help="Use test model and dictionary paths")
    parser.add_argument("-R", "--reinforced", action="store_true", help="Use the reinforced model")
    args = parser.parse_args()

    prefix = args.test
    model_type = 'reinforced' if args.reinforced else 'cnn'
    model_path = f'./weights/{prefix}_{model_type}_weights.pth' if args.test else f'./weights/{model_type}_weights.pth'
    dict_path = f'./weights/{prefix}.pkl' if args.test else './weights/label_to_letter_dict.pkl'
    
    print(f"Model path: {model_path}")
    print(f"Dictionary path: {dict_path}")
    
    # Load the label to letter dictionary from the pkl file
    with open(dict_path, 'rb') as f:
        label_to_letter_dict = pickle.load(f)

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
        for char in label_to_letter_dict.values():
            if char not in existing_characters:
                image_id = sum(len(user_images[k]) for k in user_images) + 1
                draw_letter_prompt([char], save_folder, image_id, csv_writer)
                image_path = os.path.join(save_folder, f"{image_id:04d}.png")
                image = Image.open(image_path).resize((160, 120))
                user_images[char].append(np.array(image) / 255.0)

    # Instantiate the model and load the weights
    num_classes = len(label_to_letter_dict)  # 10 digits + 26 lowercase + 26 uppercase
    model = CNNModel(num_classes=num_classes)

    # Initialize the fully connected layers
    input_shape = (1, 160, 120)
    model.initialize_fc_layers(input_shape)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))
    model.eval()

    print("Model loaded successfully!")
    reinforcement_loop(model, label_to_letter_dict, user_images, save_folder, weights_folder, prefix=prefix)
