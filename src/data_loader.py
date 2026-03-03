import tensorflow as tf
from pathlib import Path

def get_file_lists(data_dir_name = 'data/original'):
    classes = ['def_front','ok_front']
    file_paths = []
    labels = []

    current_dir = Path(__file__).parent.resolve()
    repo_root = current_dir.parent

    base_path = repo_root / data_dir_name

    for idx, label in enumerate(classes):
        class_path = base_path / label

        if not class_path.exists():
            print("Cartella non trovata.")

        images = list(class_path.glob('*.jpeg'))
        for img_path in images:
            file_paths.append(str(img_path))
            labels.append(idx)
    return file_paths, labels