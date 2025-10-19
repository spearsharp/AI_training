"""
Example experiment script demonstrating model fine-tuning workflow.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from data.data_loader import CustomDataLoader
from models.model_factory import ModelFactory
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import setup_logger
from utils.seed import set_seed
from accelerate import Accelerator


def run_bert_classification_experiment():
    """Run BERT classification experiment."""
    
    # Setup
    logger = setup_logger(name="bert-experiment", level="INFO")
    config_manager = ConfigManager("config/bert_classification.yaml")
    config = config_manager.get_config()
    
    # Set seed
    set_seed(config.training.seed)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load data
    logger.info("Loading data...")
    data_loader = CustomDataLoader(config)
    train_dataset, val_dataset, test_dataset = data_loader.load_datasets()
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model_factory = ModelFactory(config)
    model = model_factory.create_model()
    
    # Train
    trainer = Trainer(model, config, accelerator, logger)
    trainer.train(train_loader, val_loader)
    
    # Evaluate
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
        
        evaluator = Evaluator(config, logger)
        test_results = evaluator.evaluate(model, test_loader)
        logger.info(f"Test results: {test_results}")
        
        # Save results
        import json
        results_path = Path(config.training.output_dir) / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    logger.info("Experiment completed!")


if __name__ == "__main__":
    run_bert_classification_experiment()