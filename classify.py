import pickle
from model import CNNModel, CharacterModel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import sys

# Function to load the model and label dictionary
def load_character_model(model_path, dict_path, input_shape, num_classes):
    # Load the label_to_letter_dict
    with open(dict_path, 'rb') as f:
        label_to_letter_dict = pickle.load(f)

    # Initialize the model
    model = CNNModel(num_classes=num_classes)
    model.initialize_fc_layers(input_shape)

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create an instance of CharacterModel
    character_model = CharacterModel(model, label_to_letter_dict)
    return character_model

def preprocess_image(image):
    if isinstance(image, str):
        img = Image.open(image).convert("L")
    else:
        img = image.convert("L")
    img = img.resize((160, 120))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # Normalize the image
    return torch.tensor(img, dtype=torch.float32)

def classify_character(character_model, image):
    # Preprocess the image
    image = preprocess_image(image)
    # Predict the class of the image
    prediction = character_model.predict(image)
    return prediction

def main(img_path):
    # Load the model and label dictionary
    model_path = './weights/reinforced_cnn_weights.pth'
    dict_path = './weights/label_to_letter_dict.pkl'
    input_shape = (1, 160, 120)
    num_classes = 62  # 10 digits + 26 lowercase + 26 uppercase

    character_model = load_character_model(model_path, dict_path, input_shape, num_classes)

    # Load an image from the user
    try:
        image = Image.open(img_path)
    except Exception as e:
        print(f"Error: Unable to load image. {e}")
        return
    
    # Classify the character
    predicted_character = classify_character(character_model, image)
    print(f"The predicted character is: {predicted_character}")

    # Display the image
    plt.imshow(np.array(image).squeeze(), cmap="gray")
    plt.title(f"Predicted Character: {predicted_character}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python classify.py <image_path>")
    else:
        main(sys.argv[1])