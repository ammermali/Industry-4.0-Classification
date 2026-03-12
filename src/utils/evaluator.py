import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Component the evaluate the model performance on the testing dataset.

class ModelEvaluator:
    def __init__(self, target_names=['DEF', 'OK']):
        self.target_names = target_names


    def evaluate_model(self, model_path, test_ds, save_path = None):
        if not os.path.exists(model_path):
            print("Model not found")
            return

        print("Evaluating model...")
        model = tf.keras.models.load_model(model_path)

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            preds_binary = (preds > 0.5).astype(int).flatten()


            for i in range(len(labels)):
                true_label = int(labels[i].numpy())
                pred_label = int(preds_binary[i])
                score = float(preds[i][0])

                if true_label != pred_label:
                    print(f"True: {true_label} | Pred: {pred_label} | Score: {score:.4f}")

                y_true.append(true_label)
                y_pred.append(pred_label)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

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
