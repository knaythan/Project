import pickle
from model import CNNModel, CharacterModel
import numpy as np
import torch
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Function to load the model and label dictionary
def load_character_model(model_path, dict_path, input_shape):
    with open(dict_path, 'rb') as f:
        label_to_letter_dict = pickle.load(f)

    print("Loaded label_to_letter_dict")
    print(label_to_letter_dict)

    num_classes = len(label_to_letter_dict)
    model = CNNModel(num_classes=num_classes)
    model.initialize_fc_layers(input_shape)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    character_model = CharacterModel(model, label_to_letter_dict)
    return character_model

# Updated preprocessing function
def preprocess_image(image):
    if isinstance(image, str):
        img = Image.open(image).convert("L")
    else:
        img = image.convert("L")
    img = img.resize((160, 120))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # Normalize the image
    return torch.tensor(img, dtype=torch.float32)

def evaluate_model(model_path, dict_path, image_folder, csv_path, input_shape):
    character_model = load_character_model(model_path, dict_path, input_shape)
    df = pd.read_csv(csv_path)
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, row['Path'])
        true_label = row['Label']
        
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Predict the label
        predicted_character = character_model.predict(image)
        
        y_true.append(true_label)
        y_pred.append(predicted_character)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    labels = list(character_model.label_to_letter_dict.values())
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return accuracy, conf_matrix, character_model

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, labels, accuracy, reinforced=False):
    title = 'Reinforced Confusion Matrix' if reinforced else 'Pre-Reinforced Confusion Matrix'
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} (Accuracy: {accuracy:.2f})')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Character Classification Model")
    parser.add_argument("--test", type=str, nargs='?', const='lower', help="Use test model and dictionary paths")
    parser.add_argument("-R", "--reinforced", action="store_true", help="Use the reinforced model")
    args = parser.parse_args()

    prefix = args.test
    model_type = 'reinforced' if args.reinforced else 'cnn'
    model_path = f'./weights/{prefix}_{model_type}_weights.pth' if args.test else f'./weights/{model_type}_weights.pth'
    dict_path = f'./weights/{prefix}.pkl' if args.test else './weights/label_to_letter_dict.pkl'
    image_folder = './user_drawings'
    csv_path = os.path.join(image_folder, 'image_mapping.csv')
    input_shape = (1, 160, 120)

    # Evaluate the model
    accuracy, conf_matrix, character_model = evaluate_model(model_path, dict_path, image_folder, csv_path, input_shape)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    plot_confusion_matrix(conf_matrix, list(character_model.label_to_letter_dict.values()), accuracy, reinforced=args.reinforced)
    
    # Display example predictions
    df = pd.read_csv(csv_path)
    num_examples = 5
    example_indices = np.random.choice(len(df), num_examples, replace=False)

    for idx in example_indices:
        row = df.iloc[idx]
        image_path = os.path.join(image_folder, row['Path'])
        true_label = row['Label']
        
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Predict the label
        with torch.no_grad():
            predicted_character = character_model.predict(image)
        
        # Display the image and prediction
        image = Image.open(image_path).convert('L')
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_label}, Predicted: {predicted_character}")
        plt.axis('off')
        plt.show()
