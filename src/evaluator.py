import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def evaluate_model(model_path, test_ds):
    if not os.path.exists(model_path):
        print("Model not found")
        return

    print("Evaluating model...")
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose = 0)
        y_true.extend(labels.numpy())
        # Defines a treshold at 0.5
        y_pred.extend((preds > 0.5).astype(int).flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(classification_report(y_true, y_pred, target_names=['DEF', 'OK']))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted DEF','Predicted OK'],
                yticklabels=['Actual DEF','Actual OK'])

    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports\confusion_matrix.png')
    plt.show()

    return y_true, y_pred