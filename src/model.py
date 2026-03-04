import tensorflow as tf
from tensorflow.keras import layers, models

#Component that defines and builds the structure of the model.

def build_model(input_shape=(300,300,1), dropout_rate=0.3, architecture='cnn', reduction_layer='gap2d'):
    model = models.Sequential()
    match architecture:
        case 'cnn':
            model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
            model.add(layers.MaxPooling2D((2,2)))
            model.add(layers.Conv2D(64, (3,3), activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2,2)))
            match reduction_layer:
                case 'gap2d':
                    model.add(layers.GlobalAveragePooling2D())
                case 'flatten':
                    model.add(layers.Flatten())
                case 'gmp2d':
                    model.add(layers.GlobalMaxPooling2D())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(1, activation='sigmoid'))
            return model

        case 'mlp':
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(1, activation='sigmoid'))
            return model