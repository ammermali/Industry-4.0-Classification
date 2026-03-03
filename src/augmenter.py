import tensorflow as tf
from tensorflow.keras import layers

def get_augmenter():
    return tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5, fill_mode="nearest"),
        layers.RandomZoom(0.1, fill_mode="nearest")
    ])

def apply_augmentation(dataset, augmenter):
    return dataset.map(
        lambda x,y: (augmenter(x,training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )