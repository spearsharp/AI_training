#!/usr/bin/env python3
"""
Main training script for transformer fine-tuning.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from accelerate import Accelerator

from config.config_manager import ConfigManager
from data.data_loader import CustomDataLoader
from models.model_factory import ModelFactory
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.gpu_utils import get_device_info


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Fine-tuning")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="Local rank for distributed training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Setup logging
    logger = setup_logger(
        name="transformers-main",
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=config.logging.log_file if hasattr(config.logging, 'log_file') else None
    )
    
    logger.info("Starting transformer fine-tuning")
    logger.info(f"Configuration: {args.config}")
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=config.logging.tracker if hasattr(config.logging, 'tracker') else None,
    )
    
    # Device info
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Data loading
    logger.info("Loading data...")
    data_loader = CustomDataLoader(config)
    train_dataset, val_dataset, test_dataset = data_loader.load_datasets()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # Model creation
    logger.info("Creating model...")
    model_factory = ModelFactory(config)
    model = model_factory.create_model()
    
    # Training setup
    trainer = Trainer(
        model=model,
        config=config,
        accelerator=accelerator,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Training
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)
    
    # Evaluation
    if test_dataset:
        logger.info("Starting evaluation...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True
        )
        
        evaluator = Evaluator(config, logger)
        test_results = evaluator.evaluate(model, test_loader)
        logger.info(f"Test results: {test_results}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()