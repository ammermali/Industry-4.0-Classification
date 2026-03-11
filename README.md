# Industry 4.0 Cast Defect Detector

This repository provides a deep learning pipeline for classifying industrial casting products as 'OK' or 'Defective' (DEF) using various neural network architectures.

## 1. Installation

Ensure you have Python 3.9 or higher installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

After installing the requirements make sure that the folder contains both the data/ and the src/ folders.

## 2. Instruction for training

The repository allows for both training and predicting from the command line interface (CLI). 
The training start by using the following command:

```bash
python main.py --mode train --batch_size <BATCH_SIZE> --lr <LEARNING_RATE> --epochs <EPOCHS> --mp
```

You can set the parameters for batch_size, learning rate and epochs directly from this command. It's possible to activate the Mixed Precision for faster training too.

It will outputs the models/ folder, containing the trained models and the logs/ folder containing both the raw training history (CSV) and its plotting as well as the confusion matrix obtained through the evaluation of the best model.

## 3. Instruction for prediction

You can use a specific trained model to predict the class of an image by using the follow command:

```bash
python main.py --mode predict --model_path <MODEL_PATH> --image_path <IMAGE_PATH>
```

It will return the predicted class and the confidence score of the prediction.
