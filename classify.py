import torch
from PIL import Image
import sys

import torch.nn as nn
import torchvision.transforms as transforms

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

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((160, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def main(image_path):
    num_classes = 62  # Adjust based on your number of classes
    model = CNNModel(num_classes)
    model.initialize_fc_layers((1, 160, 120))

    # Load the model weights (assuming you have a saved model)
    model.load_state_dict(torch.load('./weights/79accuracy.pth', map_location=torch.device('cpu')))
    model.eval()

    image = load_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f'Predicted class: {predicted.item()}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python classify.py <image_path>")
    else:
        image_path = sys.argv[1]
        main(image_path)