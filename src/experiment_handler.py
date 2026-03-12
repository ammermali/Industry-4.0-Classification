from sklearn.model_selection import train_test_split
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.model.builder import ModelBuilder
from src.model.engine import ModelEngine
from src.utils.evaluator import ModelEvaluator
from src.utils.plotter import Plotter
from pathlib import Path


class ExperimentHandler:
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader()
        self.processor = DataProcessor(
            img_size = config.get('img_size', (300,300)),
            batch_size = config.get('batch_size', 64)
        )
        self.plotter = Plotter()
        self.evaluator = ModelEvaluator()
        self.epochs = config.get('epochs', 10)

    def run_experiments(self, experiments):
        paths, labels = self.loader.get_file_lists("data/processed/train")

        if not paths:
            print("No data found.")
            return

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        test_paths, test_labels = self.loader.get_file_lists("data/processed/test")

        train_ds = self.processor.prepare_dataset(train_paths, train_labels)
        val_ds = self.processor.prepare_dataset(val_paths, val_labels)
        test_ds = self.processor.prepare_dataset(test_paths, test_labels)

        for exp in experiments:
            model = ModelBuilder()
            model = model.build_model(
                architecture=exp['arch'],
                reduction_layer=exp['red']
            )

            model.summary()
            engine = ModelEngine(model)
            engine.compile_model(learning_rate=self.config.get('lr', 0.0001))
            engine.train(train_ds, val_ds, epochs=self.epochs, exp_name=exp['name'])

            model_path = f"models/{exp['name']}/weights_{exp['name']}.keras"
            matrix_path = f"logs/{exp['name']}/confusion_matrix.png"
            threshold_path = f"models/{exp['name']}/threshold_{exp['name']}.txt"
            self.evaluator.evaluate_model(
                model_path=model_path,
                test_ds=test_ds,
                save_path=matrix_path,
                threshold_path=threshold_path
            )
        self.plotter.plot_results(experiments)

    def run_inference(self, model_path, image_path):
        engine = ModelEngine()
        engine.load_model(model_path)

        threshold_path = model_path.replace(".keras", ".txt").replace("weights", "threshold")

        try:
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
        except FileNotFoundError:
            print(f"Threshold of {model_path} not found. Set it to 0.5.")
            threshold = 0.5

        img_tensor, _ = self.processor.load_and_resize(image_path, label = 0)

        label, score = engine.predict(img_tensor, threshold = threshold)
        print(f"Result: {label} | Score: {score}")
        return label,score

    def run_evaluation(self, model_path):
        test_paths, test_labels = self.loader.get_file_lists("data/processed/test")
        test_ds = self.processor.prepare_dataset(test_paths, test_labels)
        self.evaluator.evaluate_model(
            model_path=model_path,
            test_ds=test_ds
        )

    def run_folder_inference(self, model_path, folder_path):
        engine = ModelEngine()
        engine.load_model(model_path)

        threshold_path = model_path.replace(".keras", ".txt").replace("weights", "threshold")
        try:
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
        except FileNotFoundError:
            print(f"Threshold of {model_path} not found. Set it to 0.5.")
            threshold = 0.5
        image_paths = list(Path(folder_path).glob('*.jpeg'))
        if not image_paths:
            print(f"No .jpeg images found in {folder_path}.")
            return

        print(f"Prediction of {len(image_paths)} images. (Threshold: {threshold:.4f})...")
        results = {'DEF': 0, 'OK': 0}

        for img_path in image_paths:
            img_tensor, _ = self.processor.load_and_resize(str(img_path), label=0)
            label, score = engine.predict(img_tensor, threshold=threshold)
            results[label] += 1
        print(f"Total: {len(image_paths)}")
        print(f"OK ratio: {results['OK'] / len(image_paths)}% ({results['OK']} / {len(image_paths)})")
        print(f"DEF ratio: {results['DEF'] / len(image_paths)}% ({results['DEF']} / {len(image_paths)})")
        return results