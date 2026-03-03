import tensorflow as tf

def load_and_resize(file_path, label, img_size=(300,300)):

    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=1)
    img = tf.image.resize(img, img_size)
    return img, label

def prepare_dataset(file_paths, labels, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(lambda x,y: load_and_resize(x,y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))

    ds = ds.batch(batch_size=batch_size)

    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds