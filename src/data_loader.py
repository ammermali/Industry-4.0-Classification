import tensorflow as tf
import os

def load_and_preprocess_image(path, target_size=(512,512)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, target_size)
    image = image / 255.0

    return image

def get_dataset_paths(base_path):
    classes = ['def_front', 'ok_front']
    image_paths = []
    labels = []

    for idx, label in enumerate(classes):
        class_path = os.path.join(base_path, label)
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(idx)

    return image_paths, labels