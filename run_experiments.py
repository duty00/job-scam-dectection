"""Run the complete reproducible experiment from the project root."""
import argparse
from pathlib import Path
from modules.config import Config
from modules.experiment import run_all

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='EMSCAD: BoW, TF-IDF and pretrained GloVe comparison')
    parser.add_argument('--root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-features',type=int,default=40000)
    args=parser.parse_args()
    run_all(Config(root=args.root,seed=args.seed,max_features=args.max_features))
