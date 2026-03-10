import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

#Component that defines and builds the structure of the model.

def build_model(input_shape=(300,300,1), dropout_rate=0.5, architecture='cnn', reduction_layer='gap2d'):
    l2 = regularizers.l2(0.0001)
    model = models.Sequential()
    match architecture:
        case 'cnn':
            model.add(layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=l2))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2,2))) # 150 x 150
            model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2,2))) # 75 x 75
            model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((3,3))) # 25 x 25
            match reduction_layer:
                case 'gap2d':
                    model.add(layers.GlobalAveragePooling2D())
                case 'flatten':
                    model.add(layers.Flatten())
                case 'gmp2d':
                    model.add(layers.GlobalMaxPooling2D())
            model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(1, activation='sigmoid'))
            return model

        case 'mlp':
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(256))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(64))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            return model