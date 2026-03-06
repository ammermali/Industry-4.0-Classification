from sklearn.model_selection import train_test_split
from src.data_loader import get_file_lists
from src.data_processer import prepare_dataset
from src.data_augmenter import apply_augmentation, get_augmenter
from src.experiment_handler import run_experiment
from src.plotter import plot_models

EPOCHS = 20

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

    augmenter = get_augmenter()
    train_ds = apply_augmentation(train_ds, augmenter)

    #TODO: maybe add this to a configuration file? Like config.json
    experiments = [
        {'name': 'ModelA', 'arch': 'mlp', 'red': None, 'epochs': EPOCHS},
        {'name': 'ModelB', 'arch': 'cnn', 'red': 'gap2d', 'epochs': EPOCHS},
        {'name' : 'ModelC', 'arch': 'cnn', 'red': 'gmp2d', 'epochs': EPOCHS},
        {'name': 'ModelD', 'arch': 'cnn', 'red': 'flatten', 'epochs': EPOCHS},
    ]
    run_experiment(experiments, train_ds, val_ds, test_ds)

    plot_models(experiments)

if __name__ == "__main__":
    main()