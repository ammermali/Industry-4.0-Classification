import os
from src.model import build_model
from src.train import train
from src.evaluator import evaluate_model

def run_experiment(experiments, train_ds, val_ds, test_ds):
    for exp in experiments:
        print(f"Running experiment {exp['name']}")
        print(exp)

        model = build_model(
            architecture=exp['arch'],
            reduction_layer=exp['red']
        )

        model.summary()

        train(model, train_ds, val_ds, epochs=exp['epochs'])

        model_path = f"model/{exp['name']}.keras"
        if os.path.exists('reports/confusion_matrix.png'):
            os.rename("model/best_model.keras", model_path)

        evaluate_model_and_save(model_path, test_ds, exp['name'])

def evaluate_model_and_save(model_path, test_ds, name):
    evaluate_model(model_path, test_ds)
    if os.path.exists('reports/confusion_matrix.png'):
        os.rename("reports/confusion_matrix.png", f'reports/{name}_matrix.png')