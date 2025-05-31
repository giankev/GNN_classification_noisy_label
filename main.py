import argparse
import pandas as pd
import torch
import logging
import os

from source.config import ModelConfig
from source.trainer import ModelTrainer
from source.data_loader import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Neural Network Training and Inference')
    parser.add_argument('--test_path', required=True, help='Path to test.json.gz file')
    parser.add_argument('--train_path', help='Path to train.json.gz file (optional)')
    parser.add_argument('--model_paths_file', help='Path to file containing list of model paths for prediction (optional)')
    parser.add_argument('--model_path', help='Path to a single pretrained model .pth file for prediction (optional)')
    parser.add_argument('--num_cycles', default=10, type=int, help='Number of training cycles')
    parser.add_argument('--pretrain_paths', help='Path to pretrained model(s), comma-separated if multiple (optional)')
    parser.add_argument('--output_dir', default="./output", help='Directory to save output predictions')
    return parser.parse_args()

def main():
    args = parse_args()

    config = ModelConfig(
        test_path=args.test_path,
        train_path=args.train_path,
        num_cycles=args.num_cycles,
        pretrain_paths=args.pretrain_paths
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ModelTrainer(config, device)

    if args.train_path:
        df_train = load_dataset(args.train_path)
        trainer.train_multiple_cycles(df_train, args.num_cycles)

    else:
        if not args.model_paths_file:
            raise ValueError("Missing --train_path and --model_paths_file. You must provide at least one.")
        
        if os.path.exists(args.model_paths_file):
            with open(args.model_paths_file, 'r') as f:
                model_paths = [line.strip() for line in f.readlines()]
            trainer.models = model_paths
        else:
            raise FileNotFoundError(f"Model paths file '{args.model_paths_file}' not found.")
    
    if args.test_path:
        df_test = load_dataset(args.test_path)
        predictions, _ = trainer.predict_with_ensemble_score(df_test)

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"testset_{config.folder_name}.csv")
        
        pd.DataFrame({
            "id": range(len(predictions)),
            "pred": predictions
        }).to_csv(output_path, index=False)
        
        print(f"Predictions saved to: {output_path}")


if __name__ == '__main__':
    main()
