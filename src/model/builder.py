import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

#Component that defines and builds the structure of the model.

class ModelBuilder:
    def __init__(self, input_shape=(300,300,1), dropout_rate=0.4):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.l2 = regularizers.l2(0.0001)

    def build_model(self, architecture='cnn', reduction_layer='gap2d'):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        match architecture:
            case 'cnn':
                model.add(layers.Conv2D(32, (3,3), padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation('relu'))
                model.add(layers.MaxPooling2D((2,2)))
                model.add(layers.Conv2D(64, (3,3), padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation('relu'))
                model.add(layers.MaxPooling2D((2,2)))
                model.add(layers.Conv2D(128, (3,3), padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation('relu'))
                model.add(layers.MaxPooling2D((2,2)))
                match reduction_layer:
                    case 'gap2d':
                        model.add(layers.GlobalAveragePooling2D())
                    case 'flatten':
                        model.add(layers.Flatten())
                    case 'gmp2d':
                        model.add(layers.GlobalMaxPooling2D())
                model.add(layers.Dense(64, activation='relu', kernel_regularizer=self.l2))
                model.add(layers.Dropout(self.dropout_rate))

            case 'mlp':
                model.add(layers.Flatten())
                model.add(layers.Dense(256))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation('relu'))
                model.add(layers.Dropout(self.dropout_rate))
                model.add(layers.Dense(64))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation('relu'))
                model.add(layers.Dropout(self.dropout_rate))
                model.add(layers.Dense(16, activation='relu'))

        model.add(layers.Dense(1, activation='sigmoid', dtype='float32'))
        return model