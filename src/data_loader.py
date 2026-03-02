import tensorflow as tf
import os

def load_and_preprocess_image(path, label, target_size=(512,512)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, target_size)
    image = image / 255.0

    return image, label

def get_dataset(data_dir, batch_size = 32, is_training = True):
    classes = ['def_front', 'ok_front']
    file_paths = []
    labels = []
    for idx, label in enumerate(classes):
        class_path = os.path.join(data_dir, label)
        for filename in os.listdir(class_path):
            if filename.endswith('.jpeg'):
                file_paths.append(os.path.join(class_path, filename))
                labels.append(idx)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(lambda p, l: load_and_preprocess_image(p, l), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(buffer_size=len(file_paths))

    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds