# Human Reinforcement Learning Model

This project implements a human reinforcement learning model using PyTorch. The model prompts the user to draw characters and improves its accuracy based on user input.

## Setup Instructions
1. Extract the Img.7z file into the same folder as the Jupyter Notebooks.
2. Ensure the CSV file is named `labels.csv`.
3. Verify that the Img folder contains all the image files.
4. Install the required dependencies using:
```bash
    python -m pip install -r requirements.txt
```
5. Ensure your Jupyter Notebook is using the Python version that you installed the dependencies to.

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
    git clone https://github.com/knaythan/Project.git
```
2. Navigate to the project directory:
```bash
    cd human-rl-model
```

### Usage

For all the commands below:
```bash
python <file_name>.py <arguments> -R
```
Make sure that the weight is named `reinforced_weights.pth` and the mapping file is named `label_to_letter_dict.pkl`

To run specific models rename them with a prefix `prefix_reinforced_weights.pth` and `prefix.pkl` and specify it using:
```bash
python classify.py <arguments> --test <prefix>
```
By default the file uses the lowercase model for testing.

To run a specific reinforced model, use:
```bash
python <file_name>.py <arguments> -R --test
```
This will run `prefix_reinforced_weights.pth` with `prefix.pth`. Make sure that the file is named properly after using the notebook, or the code will not work.

## Reinforcement
If you want to train your own model, extract the zip files as mentioned in the setup instructions.

To use the provided weights and perform reinforcement learning, simply run:
```bash
python reinforcement.py
```

You can also run a smaller test with just lowercase letters by running:
```bash
python reinforcement.py --test
```

All images drawn by the user for reinforcement are saved in `/user_drawings`, with each image-to-label mapping stored in `/user_drawings/image_mapping.csv`.

## Classification
After performing reinforcement learning, you can classify an image by using:
```bash
python classify.py --image /path/to/image
```

To create a canvas for the user to draw an image and classify it, run:
```bash
python classify.py --user-draw
```

## Evaluation
After training the model, you can use evaluate.py to see a confusion matrix and examples of predictions by a model on the user's drawings:
```bash
python evaluate.py
```