from sklearn.model_selection import train_test_split
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.model.builder import ModelBuilder
from src.model.engine import ModelEngine
from src.utils.evaluator import ModelEvaluator
from src.utils.plotter import Plotter
import os


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

            engine = ModelEngine(model)
            engine.compile_model(learning_rate=self.config.get('lr', 0.0001))
            engine.train(train_ds, val_ds, epochs=self.epochs, exp_name=exp['name'])

            model_path = f"models/{exp['name']}.keras"
            matrix_path = f"logs/{exp['name']}/confusion_matrix.png"
            self.evaluator.evaluate_model(
                model_path=model_path,
                test_ds=test_ds,
                save_path=matrix_path
            )
        self.plotter.plot_results(experiments)

    def run_inference(self, model_path, image_path):
        engine = ModelEngine()
        engine.load_model(model_path)

        img_tensor, _ = self.processor.load_and_resize(image_path, label = 0)

        label, score = engine.predict(img_tensor)
        print(f"Result: {label} | Score: {score}")
        return label,score