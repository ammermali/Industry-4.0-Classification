import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics

# The only responsibility of this component is to converge the loss function.

def train(model, train_ds, val_ds, epochs=10, learning_rate=0.001):
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
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