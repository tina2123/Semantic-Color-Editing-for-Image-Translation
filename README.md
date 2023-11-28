# Semantic-Color-Editing-for-Image-Translation
 Pretraining is All You Need with Semantic Color Editing for Image-to-Image Translation
 ## Introduction
This repository, part of the coursework for MA-INF 2307 - Lab Computer Vision, enhances the image-to-image translation task by introducing Semantic Color Editing. Users can select specific categories within an image and choose to color them with shades of green, red, or blue. The functionality is demonstrated through a user-friendly Gradio interface, as detailed in the [lab report]([https://github.com/PITI-Synthesis/PITI](https://drive.google.com/file/d/18FduahKdY7O4SHAE7ZFwDJfoDbbt3QVh/view?usp=sharing)).

## Installation
```bash
git clone git@github.com:tina2123/Semantic-Color-Editing-for-Image-Translation.git
cd PITI
```

## Environment
```bash
conda env create -f environment.yml
```

## Usage
To run the Semantic Color Editing interface, execute the following command:
```bash
python inference.py
```

After running the command, navigate to the Gradio interface URL that will be displayed in your terminal. Here's how to use the interface:
1. Upload an image to the interface.
2. Select the category you want to edit from the label.txt file listed in the interface.
3. Choose your desired color tone: greenish, reddish, or blueish.
The labels.txt contains all the categories that the model can recognize and edit.


## Example


## Acknowledgments
Thanks to the creators of the [PITI](https://github.com/PITI-Synthesis/PITI) repository for the initial image translation model.
MA-INF 2307 - Lab Computer Vision for the support and guidance.
