import os
from src.model import build_model
from src.train import train
from src.evaluator import evaluate_model

def run_experiment(experiments, train_ds, val_ds, test_ds, class_weights=None):
    for exp in experiments:
        print(f"Running experiment {exp['name']}")
        print(exp)

        model = build_model(
            architecture=exp['arch'],
            reduction_layer=exp['red']
        )

        model.summary()

        train(model,
              train_ds,
              val_ds,
              epochs=exp['epochs'],
              class_weights=class_weights,
              exp_name=exp['name']
              )

        model_path = f"model/{exp['name']}.keras"
        matrix_save_path = f"logs/{exp['name']}/confusion_matrix.png"
        evaluate_model(model_path, test_ds, save_path=matrix_save_path)