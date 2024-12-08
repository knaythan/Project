# Human Reinforcement Learning Model

This project implements a human reinforcement learning model using PyTorch. The model prompts the user to draw characters and improves its accuracy based on user input.

## Setup Instructions
1. Extract the Img.7z file into the same folder as the Jupyter Notebooks
2. Make sure that the name of the csv is labels.csv
3. Make sure the Img file contains all of the image files
4. Use `python -m pip install -r requirements.txt` to install the correct dependencies
5. Ensure your notebook is using the python version that you pip installed to

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip

### Installation

1. Clone the repository: 

### Usage

If you want to train your own model, extract the zip files as mentioned in the setup instructions. 

To use the provided weights and perform reinforcement learning, simply run:
```bash
python reinforcement.py
```

After performing reinforcement learning, you can classify an image by using:
```bash
python classify.py <path to image to classify>
```