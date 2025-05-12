# Model Directory

This directory is intended to store trained machine learning models for sign language recognition.

## Expected Files

- `keras_model.h5`: A trained Keras model for sign language classification
- `labels.txt`: A text file containing the labels for the classes the model can recognize

## Training Your Own Model

To train your own model:

1. Use the `datacollection.py` script to collect images for each sign language gesture
2. Use a tool like [Teachable Machine](https://teachablemachine.withgoogle.com/) to train a model
3. Export the model as a Keras model
4. Place the model files in this directory

## Note

The current implementation uses skin detection instead of a trained model for hand detection. When you have a trained model, you can modify the `test.py` file to use it for classification.
