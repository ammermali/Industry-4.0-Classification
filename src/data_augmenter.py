import tensorflow as tf
from tensorflow.keras import layers

# Component that defines and applies the augmentation.

def get_augmenter():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2, fill_mode="nearest"),
        layers.RandomZoom(0.2, fill_mode="nearest"),
        layers.RandomContrast(factor=0.2)
    ])

def apply_augmentation(dataset, augmenter):
    augmented_ds = dataset.map(
        lambda x, y: (augmenter(x,training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.concatenate(augmented_ds)