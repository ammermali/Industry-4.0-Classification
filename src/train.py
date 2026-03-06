import os
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
import time

class EpochTimer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = time.time() - self.start_time

# The only responsibility of this component is to converge the loss function.

def train(model, train_ds, val_ds, epochs=10, learning_rate=0.0001, exp_name="best_model"):
    os.makedirs('models', exist_ok=True)
    os.makedirs(f'logs/{exp_name}', exist_ok=True)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )

    run_callbacks = [
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
            patience=2,
            min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(
            f'logs/{exp_name}/history.csv',
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=run_callbacks,
        verbose=1
    )

    return history