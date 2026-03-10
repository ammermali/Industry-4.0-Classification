import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import sys
from src.experiment_handler import ExperimentHandler

 # This component defines an entry point for the whole system. Both for training and inference.

def main():

    # Args
    parser = argparse.ArgumentParser(description="Industry 4.0 Cast Defect Detector - Training and Inference")
    parser.add_argument(
        '--mode',
        choices=['train', 'predict'],
        default='train',
        help='Mode: train or predict'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        help="Path of the chosen model for inference."
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help="Path of the image to predict."
    )
    args = parser.parse_args()


    # Config
    config = {
        'batch_size': 64,
        'lr': 0.001, # learning rate
        'img_size': (300,300),
        'epochs': 15
    }

    # Experiments
    experiments = [
        {'name': 'ModelA', 'arch': 'mlp', 'red': None},
        {'name': 'ModelB', 'arch': 'cnn', 'red': 'gap2d'},
        {'name' : 'ModelC', 'arch': 'cnn', 'red': 'gmp2d'},
        {'name': 'ModelD', 'arch': 'cnn', 'red': 'flatten'},

    ]


    handler = ExperimentHandler(config)

    if args.mode == "train":
        print(f"Starting training for {len(experiments)} experiments.")
        handler.run_experiments(experiments)

    elif args.mode == 'predict':
        if not args.model_path or not args.image_path:
            parser.error(f"Model_path and image_path are required.")
        handler.run_experiments(args.model_path, args.image_path)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(e)
        sys.exit(1)