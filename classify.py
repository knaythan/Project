import pickle
from model import CNNModel, CharacterModel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import argparse
import tkinter as tk
from PIL import ImageGrab
import os
import csv

# Function to load the model and label dictionary
def load_character_model(model_path, dict_path, input_shape):
    # Load the label_to_letter_dict
    with open(dict_path, 'rb') as f:
        label_to_letter_dict = pickle.load(f)
        
    print("Loaded label_to_letter_dict")
    print(label_to_letter_dict)
    
    num_classes = len(label_to_letter_dict)

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

def main(img_path, model_path, dict_path):
    input_shape = (1, 160, 120)

    character_model = load_character_model(model_path, dict_path, input_shape)

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

def draw_and_classify(character_model):
    root = tk.Tk()
    root.state('zoomed')
    canvas = tk.Canvas(root, width=1200, height=900, bg="white")
    canvas.pack()

    def paint(event):
        x1, y1 = (event.x - 30), (event.y - 30)
        x2, y2 = (event.x + 30), (event.y + 30)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=1)

    def classify():
        canvas.update()
        x = canvas.winfo_rootx() + 175
        y = canvas.winfo_rooty()
        x1 = x + canvas.winfo_width() + 175
        y1 = y + canvas.winfo_height() + 150
        image = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")
        border_crop = 5
        width, height = image.size
        cropped_image = image.crop((border_crop, border_crop, width - border_crop, height - border_crop))
        resized_image = cropped_image.resize((160, 120))
        predicted_character = classify_character(character_model, resized_image)
        print(f"The predicted character is: {predicted_character}")
        plt.imshow(np.array(resized_image).squeeze(), cmap="gray")
        plt.title(f"Predicted Character: {predicted_character}")
        plt.show()

    def clear():
        canvas.delete("all")

    def exit_program():
        print("Exiting the program.")
        exit()
        root.destroy()

    canvas.bind("<B1-Motion>", paint)
    button_frame = tk.Frame(root)
    button_frame.pack()
    classify_button = tk.Button(button_frame, text="Classify", command=classify)
    classify_button.pack(side="left", padx=5)
    clear_button = tk.Button(button_frame, text="Clear", command=clear)
    clear_button.pack(side="left", padx=5)
    exit_button = tk.Button(button_frame, text="Exit", command=exit_program)
    exit_button.pack(side="left", padx=5)
    root.protocol("WM_DELETE_WINDOW", exit_program)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character Classification")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--test", type=str, nargs='?', const='lower', help="Use test model and dictionary paths")
    parser.add_argument("-U", "--user-draw", action="store_true", help="Create a canvas for the user to draw an image")
    parser.add_argument("-R", "--reinforced", action="store_true", help="Use the reinforced model")
    args = parser.parse_args()

    
    prefix = args.test
    model_type = 'reinforced' if args.reinforced else 'cnn'
    model_path = f'./weights/{prefix}_{model_type}_weights.pth' if args.test else f'./weights/{model_type}_weights.pth'
    dict_path = f'./weights/{prefix}.pkl' if args.test else './weights/label_to_letter_dict.pkl'
        
    if args.user_draw:
        input_shape = (1, 160, 120)
        character_model = load_character_model(model_path, dict_path, input_shape)
        draw_and_classify(character_model)
    else:
        if not args.image:
            print("Error: --image_path is required if --user-draw is not specified.")
            exit(1)
        main(args.image, model_path, dict_path)
