import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import get_file_lists
from src.data_processer import prepare_dataset
from src.experiment_handler import run_experiment

def main():
    paths, labels = get_file_lists("data/processed/train")
    if not paths:
        return
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    test_paths, test_labels = get_file_lists("data/processed/test")

    train_ds = prepare_dataset(train_paths, train_labels, batch_size=32, shuffle=True)
    val_ds = prepare_dataset(val_paths, val_labels, batch_size=32, shuffle=False)
    test_ds = prepare_dataset(test_paths, test_labels, batch_size=32, shuffle=False)

    #TODO: maybe this is better in the processer component
    neg, pos = np.bincount(labels)
    total = neg + pos
    class_weight = {
        0: (1 / neg) * (total / 2.0),
        1: (1 / pos) * (total / 2.0)
    }

    #TODO: maybe add this to a configuration file? Like config.json
    experiments = [
        {'name': 'ModelA', 'arch': 'mlp', 'red': None, 'epochs': 10},
        {'name': 'ModelB', 'arch': 'cnn', 'red': 'gap2d', 'epochs': 10},
        {'name' : 'ModelC', 'arch': 'cnn', 'red': 'gmp2d', 'epochs': 10},
        {'name': 'ModelD', 'arch': 'cnn', 'red': 'flatten', 'epochs': 10},
    ]
    run_experiment(experiments, train_ds, val_ds, test_ds)

if __name__ == "__main__":
    main()