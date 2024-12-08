import torch
import torch.nn as nn

# Enhanced CNN
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.num_classes = num_classes

        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x160x120
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x80x60

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x80x60
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x40x30

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x40x30
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x20x15

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: 256x20x15
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256x10x7

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Output: 512x10x7
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 512x5x3

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
    
# Export classes
class CharacterModel:
    def __init__(self, model, label_to_letter_dict):
        self.model = model
        self.label_to_letter_dict = label_to_letter_dict

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            output = self.model(image)
            _, predicted_label = torch.max(output, 1)
            predicted_character = self.label_to_letter_dict[predicted_label.item()]
        return predicted_character