import os
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
import time

class EpochTimer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = time.time() - self.start_time

class ModelEngine:
    def __init__(self, model=None):
        self.model = model

    def compile_model(self, learning_rate=0.0001):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=0.1),
            loss=losses.BinaryCrossentropy(),
            metrics=[
                metrics.BinaryAccuracy(name='accuracy'),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall')
            ]
        )

    def train(self, train_ds, val_ds, epochs=10, exp_name="Model"):
        os.makedirs(f'models/{exp_name}', exist_ok=True)
        os.makedirs(f'logs/{exp_name}', exist_ok=True)

        callbacks = [
            EpochTimer(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'models/{exp_name}.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=7,
                min_lr=1e-7
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.CSVLogger(f'logs/{exp_name}/history.csv'),
        ]

        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img_tensor):
        img_array = tf.expand_dims(img_tensor, 0)
        prediction = self.model.predict(img_array, verbose=0)
        score = prediction[0][0]
        label = "OK" if score > 0.5 else "DEF"
        return label, score