from src.data_loader import get_file_lists
from src.data_processer import prepare_dataset
from src.data_augmenter import get_augmenter, apply_augmentation
from src.model import build_model
from src.train import train
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    paths, labels = get_file_lists("data/processed/train")

    if not paths:
        print("No data found")
        return

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = prepare_dataset(train_paths, train_labels, batch_size=32, shuffle=True)
    val_ds = prepare_dataset(val_paths, val_labels, batch_size=32, shuffle=False)

    #Data Augmentation
    #Opzionale

    #augmenter = get_augmenter()
    #train_ds = apply_augmentation(train_ds, augmenter)

    neg, pos = np.bincount(labels)
    total = neg + pos
    weight_for_def = (1 / neg) * (total / 2.0)
    weight_for_ok = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_def, 1: weight_for_ok}

    model = build_model(input_shape=(300,300,1))
    model.summary()

    history = train(
        model,
        train_ds,
        val_ds,
        epochs=20,
        learning_rate=0.001
    )

if __name__ == "__main__":
    main()