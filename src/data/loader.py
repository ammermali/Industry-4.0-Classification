from pathlib import Path

# Component that handles the data loading.

class DataLoader:
    def __init__(self):
        self.classes = ['def_front', 'ok_front']

    def get_file_lists(self, data_dir_name):
        file_paths = []
        labels = []

        current_dir = Path(__file__).parent.parent.resolve()
        repo_root = current_dir.parent
        base_path = repo_root / data_dir_name

        for idx, label in enumerate(self.classes):
            class_path = base_path / label

            if not class_path.exists():
                print("Folder not found.")

            images = list(class_path.glob('*.jpeg'))
            for img_path in images:
                file_paths.append(str(img_path))
                labels.append(idx)
        return file_paths, labels