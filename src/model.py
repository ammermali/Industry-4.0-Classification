import tensorflow as tf
from tensorflow.keras import layers, models

#Component that defines and builds the structure of the model.

def build_model(input_shape=(512,512,1)):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1,activation='sigmoid')
        ]
    )

    return model