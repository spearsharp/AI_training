#!/usr/bin/env python3
"""
Standalone evaluation script.
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.config_manager import ConfigManager
from data.data_loader import CustomDataLoader
from evaluation.evaluator import Evaluator
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, default=None, help="Path to test data")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", 
                       help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(name="evaluation", level="INFO")
    logger.info("Starting evaluation...")
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Override batch size if specified
    if args.batch_size:
        config.training.eval_batch_size = args.batch_size
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    if config.model.type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    else:
        # Add support for other model types
        raise ValueError(f"Model type {config.model.type} not supported in this script")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load data
    logger.info("Loading evaluation data...")
    data_loader = CustomDataLoader(config)
    
    if args.data_path:
        # Load custom data
        # Implement custom data loading
        raise NotImplementedError("Custom data loading not yet implemented")
    else:
        # Use test split from config
        _, _, test_dataset = data_loader.load_datasets()
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Evaluate
    evaluator = Evaluator(config, logger)
    results = evaluator.evaluate(model, test_loader, tokenizer)
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()