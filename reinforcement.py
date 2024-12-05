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

# Enhanced CNN
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.num_classes = num_classes

        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x320x240
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x160x120

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x160x120
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x80x60

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x80x60
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x40x30

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: 256x40x30
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256x20x15

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Output: 512x20x15
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 512x10x7

        # Fully connected layers are initialized dynamically
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def initialize_fc_layers(self, input_shape):
        """Initialize the fully connected layers dynamically based on the input shape."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool1(self.bn1(self.conv1(dummy_input)))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            x = self.pool5(self.conv5(x))
            flattened_size = x.numel()  # Total elements after flattening

        # Dynamically initialize fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        # Print the initialized sizes for debugging
        print(f"Initialized fc1 with input size {flattened_size} and output size 512")
        print(f"Initialized fc2 with input size 512 and output size 256")
        print(f"Initialized fc3 with input size 256 and output size {self.num_classes}")

    def forward(self, x):
        # Convolutional blocks
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

# Drawing prompt function
def draw_letter_prompt(letter, save_folder, image_id):
    os.makedirs(save_folder, exist_ok=True)
    root = tk.Tk()
    root.title(f"Draw the Letter: {letter}")
    canvas = tk.Canvas(root, width=640, height=480, bg="white")
    canvas.pack()

    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)

    def save():
        # Get the canvas's exact bounding box relative to the screen
        canvas.update()
        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
        
        # Capture the exact canvas area
        image = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")
        
        # Resize and save the image
        image = image.resize((160, 120))
        image_path = os.path.join(save_folder, f"{image_id:04d}.png")
        image.save(image_path)
        print(f"Saved image to {image_path}")
        root.destroy()

    def clear():
        canvas.delete("all")

    canvas.bind("<B1-Motion>", paint)

    button_frame = tk.Frame(root)
    button_frame.pack()

    save_button = tk.Button(button_frame, text="Save", command=save)
    save_button.pack(side="left", padx=5)

    clear_button = tk.Button(button_frame, text="Clear", command=clear)
    clear_button.pack(side="left", padx=5)

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


def reinforcement_loop(model, characters, user_images, save_folder, num_epochs=5, batch_size=16, accuracy_threshold=0.85):
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    transform = ToTensor()

    for char in characters:
        while True:
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

            # Compute Accuracy
            accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
            print(f"Accuracy for '{char}': {accuracy:.2f}%")

            # Check if accuracy is sufficient
            if accuracy >= 85.0:
                print(f"Accuracy for '{char}' is above 85%. Moving to the next character.")
                break

            # If accuracy is below 85%, collect more data
            print(f"Accuracy for '{char}' is below 85.00%. Collecting more data.")
            new_image_id = sum(len(user_images[k]) for k in user_images) + 1
            draw_letter_prompt(char, save_folder, new_image_id)
            image_path = os.path.join(save_folder, f"{new_image_id:04d}.png")
            image = Image.open(image_path).resize((160, 120)).convert('L')
            user_images[char].append(np.array(image) / 255.0)

    print("All characters have been trained and evaluated!")




# Main
if __name__ == "__main__":
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    save_folder = "user_drawings"
    user_images = defaultdict(list)

    if not os.path.exists(os.path.join(save_folder, "image_mapping.csv")):
        for char in characters:
            image_id = len(user_images) + 1
            draw_letter_prompt(char, save_folder, image_id)
            image_path = os.path.join(save_folder, f"{image_id:04d}.png")
            image = Image.open(image_path).resize((160, 120))
            user_images[char].append(np.array(image) / 255.0)
        with open(os.path.join(save_folder, "image_mapping.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Path", "Label"])
            for char, imgs in user_images.items():
                for img_id, _ in enumerate(imgs, 1):
                    writer.writerow([f"{img_id:04d}.png", char])
    else:
        with open(os.path.join(save_folder, "image_mapping.csv"), "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                path, char = row
                image = Image.open(os.path.join(save_folder, path)).resize((160, 120))
                user_images[char].append(np.array(image) / 255.0)

    # Instantiate the model and load the weights
    num_classes = 62  # 10 digits + 26 lowercase + 26 uppercase
    model = CNNModel(num_classes=num_classes)

    # Initialize the fully connected layers
    input_shape = (1, 160, 120)
    model.initialize_fc_layers(input_shape)

    weights_path = './weights/79accuracy.pth'
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    print("Model loaded successfully!")
    reinforcement_loop(model, characters, user_images, save_folder)

