import tensorflow as tf

# Component that handles the preprocessing, normalization and augmentation of the data.

class DataProcessor():
    def __init__(self, img_size=(300,300), batch_size=64):
        self.img_size = img_size
        self.batch_size = batch_size

    def load_and_resize(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, self.img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def get_augmenter(self):
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.05, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.05, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="reflect"
            )
        ])

    def prepare_dataset(self, file_paths, labels, augment=True, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds = ds.map(
            lambda x,y: self.load_and_resize(x,y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if augment:
            augmenter = self.get_augmenter()
            ds = ds.map(
                lambda x,y: (augmenter(x, training=True), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        if shuffle:
            ds = ds.shuffle(buffer_size=len(file_paths))

        ds = ds.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds