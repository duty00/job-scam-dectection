"""Train the optional TextCNN extension on the same locked EMSCAD split."""
import argparse
from pathlib import Path

from modules.config import Config
from modules.deep_learning import run_textcnn
from modules.experiment import prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EMSCAD TextCNN deep-learning extension')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    context = prepare(Config(root=args.root, seed=args.seed))
    print(run_textcnn(context, epochs=args.epochs, batch_size=args.batch_size).to_string(index=False))
