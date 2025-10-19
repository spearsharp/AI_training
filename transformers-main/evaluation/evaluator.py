"""
Main evaluator for model evaluation.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import numpy as np
from tqdm import tqdm
import logging

from .metrics import MetricsCalculator


logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator for transformer models."""
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator(config)
        
    def evaluate(self, model, dataloader: DataLoader, 
                tokenizer = None) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            tokenizer: Tokenizer (needed for generation tasks)
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Starting evaluation...")
        
        model.eval()
        device = next(model.parameters()).device
        
        all_predictions = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Collect loss
                if hasattr(outputs, 'loss'):
                    all_losses.append(outputs.loss.item())
                
                # Collect predictions and labels
                if self.config.model.type == "classification":
                    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
                    
                elif self.config.model.type in ["generation", "seq2seq"]:
                    # For generation tasks, we need to decode the outputs
                    if tokenizer:
                        generated_ids = model.generate(
                            batch['input_ids'],
                            max_length=getattr(self.config.model, 'max_length', 512),
                            num_beams=getattr(self.config.model.generation, 'num_beams', 1) if hasattr(self.config.model, 'generation') else 1,
                            temperature=getattr(self.config.model.generation, 'temperature', 1.0) if hasattr(self.config.model, 'generation') else 1.0,
                        )
                        
                        predictions_text = tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )
                        labels_text = tokenizer.batch_decode(
                            batch['labels'], skip_special_tokens=True
                        )
                        
                        all_predictions.extend(predictions_text)
                        all_labels.extend(labels_text)
        
        # Compute metrics
        metrics = {}
        
        if all_predictions and all_labels:
            metrics = self.metrics_calculator.compute_metrics(
                all_predictions, all_labels
            )
        
        # Add loss metrics
        if all_losses:
            metrics['eval_loss'] = np.mean(all_losses)
            
            # Compute perplexity for generation tasks
            if self.config.model.type in ["generation", "seq2seq"]:
                metrics['eval_perplexity'] = self.metrics_calculator.compute_perplexity(all_losses)
        
        self.logger.info(f"Evaluation completed. Metrics: {metrics}")
        
        return metrics
    
    def evaluate_and_save(self, model, dataloader: DataLoader,
                         output_path: str, tokenizer = None) -> Dict[str, Any]:
        """
        Evaluate model and save results to file.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            output_path: Path to save results
            tokenizer: Tokenizer (needed for generation tasks)
            
        Returns:
            Dictionary of evaluation results
        """
        results = self.evaluate(model, dataloader, tokenizer)
        
        # Save results
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        
        return results
    
    def predict(self, model, dataloader: DataLoader,
               tokenizer = None) -> List[Any]:
        """
        Generate predictions without computing metrics.
        
        Args:
            model: Model to use for prediction
            dataloader: DataLoader with data
            tokenizer: Tokenizer (needed for generation tasks)
            
        Returns:
            List of predictions
        """
        model.eval()
        device = next(model.parameters()).device
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.config.model.type == "classification":
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_predictions.extend(predictions)
                    
                elif self.config.model.type in ["generation", "seq2seq"]:
                    if tokenizer:
                        generated_ids = model.generate(
                            batch['input_ids'],
                            max_length=getattr(self.config.model, 'max_length', 512),
                        )
                        
                        predictions_text = tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )
                        all_predictions.extend(predictions_text)
        
        return all_predictions