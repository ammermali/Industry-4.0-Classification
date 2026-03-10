import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import sys
from src.experiment_handler import ExperimentHandler
from tensorflow.keras import mixed_precision

 # This component defines an entry point for the whole system. Both for training and inference.

def main():
    # Experiments
    experiments = [
        {'name': 'ModelA', 'arch': 'mlp', 'red': None},
        {'name': 'ModelB', 'arch': 'cnn', 'red': 'gap2d'},
        {'name': 'ModelC', 'arch': 'cnn', 'red': 'gmp2d'},
        {'name': 'ModelD', 'arch': 'cnn', 'red': 'flatten'},
    ]

    # Args
    parser = argparse.ArgumentParser(description="Industry 4.0 Cast Defect Detector - Training and Inference")
    parser.add_argument(
        '--mode',
        choices=['train', 'predict'],
        default='train',
        help='Mode: train or predict'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training mode.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate for training mode.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='No. of epochs for training mode.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Path of the chosen model for inference. Only for predict mode."
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help="Path of the image to predict. Only for predict mode."
    )
    parser.add_argument(
        "--mp",
        action="store_true",
        help="Activate the Mixed Precision. Only for training mode.")
    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'img_size': (300,300),
        'epochs': args.epochs
    }

    handler = ExperimentHandler(config)


    if args.mode == "train":
        print(f"Starting training for {len(experiments)} experiments.")
        handler.run_experiments(experiments)
        if args.amp:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("Training with mixed precision.")

    elif args.mode == 'predict':
        if not args.model_path or not args.image_path:
            parser.error(f"Model_path and image_path are required.")
        handler.run_inference(args.model_path, args.image_path)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(e)
        sys.exit(1)