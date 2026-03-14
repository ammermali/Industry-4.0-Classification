import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import os

# Component the evaluate the model performance on the testing dataset.

class ModelEvaluator:
    def __init__(self, target_names=['DEF', 'OK']):
        self.target_names = target_names

    def find_optimal_threshold(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j = tpr - fpr
        best_idx = np.argmax(j)
        return thresholds[best_idx]

    def evaluate_model(self, model_path, test_ds, save_path = None, threshold_path = None):
        if not os.path.exists(model_path):
            print("Model not found")
            return

        print("Evaluating model...")
        model = tf.keras.models.load_model(model_path)

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_pred.extend(preds.flatten())
            y_true.extend(labels.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        threshold = self.find_optimal_threshold(y_true, y_pred)

        print(threshold)

        if threshold_path:
            with open(threshold_path, 'w') as f:
                f.write(str(threshold))

        y_pred = (y_pred > threshold).astype(int)

        print(classification_report(y_true, y_pred, target_names=['DEF', 'OK'], zero_division=0))

        if save_path:
            matrix = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted DEF', 'Predicted OK'],
                        yticklabels=['Actual DEF', 'Actual OK'])

            plt.title('Confusion Matrix')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
            plt.close()

        return y_true, y_pred
