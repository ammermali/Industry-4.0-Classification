import tensorflow as tf
import os

def predict_image(model_path, image_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found.")

    model = tf.keras.models.load_model(model_path)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.cast(img, tf.float32) / 127.5 - 1
    img_array = tf.expand_dims(img, 0)
    prediction = model.predict(img_array, verbose = 0)
    score = prediction[0][0]
    if score > 0.5:
        return "OK", score
    else:
        return "DEF", score