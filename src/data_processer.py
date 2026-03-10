import tensorflow as tf

def load_and_resize(file_path, label, img_size=(300,300)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 127.5 - 1 # [-1,1]
    return img, label
def prepare_dataset(file_paths, labels, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(lambda x,y: load_and_resize(x,y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds