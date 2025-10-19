"""
Evaluation metrics calculation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import logging

try:
    from datasets import load_metric
    from evaluate import load as load_evaluate_metric
    HF_EVALUATE_AVAILABLE = True
except ImportError:
    HF_EVALUATE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Hugging Face evaluate library not available. Some metrics may not work.")


class MetricsCalculator:
    """Calculate various evaluation metrics for different tasks."""
    
    def __init__(self, config):
        self.config = config
        self.task_type = config.model.type
        
        # Load Hugging Face metrics if available
        self.hf_metrics = {}
        if HF_EVALUATE_AVAILABLE and hasattr(config.evaluation, 'metrics'):
            self._load_hf_metrics()
    
    def _load_hf_metrics(self):
        """Load Hugging Face metrics."""
        try:
            for metric_name in self.config.evaluation.metrics:
                if metric_name in ['bleu', 'rouge', 'meteor', 'bertscore']:
                    self.hf_metrics[metric_name] = load_evaluate_metric(metric_name)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not load HF metrics: {e}")
    
    def compute_metrics(self, predictions: Union[List, np.ndarray], 
                       labels: Union[List, np.ndarray],
                       texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute metrics based on task type.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            texts: Original texts (for generation tasks)
            
        Returns:
            Dictionary of computed metrics
        """
        if self.task_type == "classification":
            return self._compute_classification_metrics(predictions, labels)
        elif self.task_type == "generation":
            return self._compute_generation_metrics(predictions, labels, texts)
        elif self.task_type == "seq2seq":
            return self._compute_seq2seq_metrics(predictions, labels, texts)
        else:
            return {}
    
    def _compute_classification_metrics(self, predictions: np.ndarray, 
                                      labels: np.ndarray) -> Dict[str, float]:
        """Compute metrics for classification tasks."""
        metrics = {}
        
        # Convert to numpy arrays if needed
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # Handle multi-class vs binary classification
        num_classes = len(np.unique(labels))
        average = self.config.evaluation.get('average', 'weighted')
        
        if num_classes > 2:
            metrics['precision'] = precision_score(labels, predictions, average=average, zero_division=0)
            metrics['recall'] = recall_score(labels, predictions, average=average, zero_division=0)
            metrics['f1'] = f1_score(labels, predictions, average=average, zero_division=0)
        else:
            # Binary classification
            metrics['precision'] = precision_score(labels, predictions, zero_division=0)
            metrics['recall'] = recall_score(labels, predictions, zero_division=0)
            metrics['f1'] = f1_score(labels, predictions, zero_division=0)
            
            # AUC for binary classification if probabilities available
            # This would require prediction probabilities, not just predictions
        
        # Per-class metrics (optional)
        if hasattr(self.config.evaluation, 'detailed_metrics') and self.config.evaluation.detailed_metrics:
            metrics.update(self._compute_detailed_classification_metrics(predictions, labels))
        
        return metrics
    
    def _compute_generation_metrics(self, predictions: List[str], 
                                  references: List[str],
                                  texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute metrics for text generation tasks."""
        metrics = {}
        
        # Perplexity (if available from model outputs)
        # This would typically be computed during model inference
        
        # BLEU score
        if 'bleu' in self.hf_metrics:
            try:
                bleu_result = self.hf_metrics['bleu'].compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                metrics['bleu'] = bleu_result['bleu']
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not compute BLEU: {e}")
        
        # ROUGE scores
        if 'rouge' in self.hf_metrics:
            try:
                rouge_result = self.hf_metrics['rouge'].compute(
                    predictions=predictions,
                    references=references
                )
                for key, value in rouge_result.items():
                    metrics[f'rouge_{key}'] = value
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not compute ROUGE: {e}")
        
        return metrics
    
    def _compute_seq2seq_metrics(self, predictions: List[str], 
                                references: List[str],
                                texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute metrics for sequence-to-sequence tasks."""
        metrics = {}
        
        # BLEU score
        if 'bleu' in self.hf_metrics:
            try:
                bleu_result = self.hf_metrics['bleu'].compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                metrics['bleu'] = bleu_result['bleu']
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not compute BLEU: {e}")
        
        # ROUGE scores
        if 'rouge' in self.hf_metrics:
            try:
                rouge_result = self.hf_metrics['rouge'].compute(
                    predictions=predictions,
                    references=references
                )
                # Extract specific ROUGE scores
                if hasattr(self.config.evaluation, 'rouge_types'):
                    for rouge_type in self.config.evaluation.rouge_types:
                        if rouge_type in rouge_result:
                            metrics[f'rouge_{rouge_type}'] = rouge_result[rouge_type]
                else:
                    for key, value in rouge_result.items():
                        metrics[f'rouge_{key}'] = value
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not compute ROUGE: {e}")
        
        # METEOR score
        if 'meteor' in self.hf_metrics:
            try:
                meteor_result = self.hf_metrics['meteor'].compute(
                    predictions=predictions,
                    references=references
                )
                metrics['meteor'] = meteor_result['meteor']
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not compute METEOR: {e}")
        
        return metrics
    
    def _compute_detailed_classification_metrics(self, predictions: np.ndarray, 
                                               labels: np.ndarray) -> Dict[str, Any]:
        """Compute detailed classification metrics."""
        metrics = {}
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        
        # Add per-class metrics
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                for metric_name, value in class_metrics.items():
                    metrics[f'{class_name}_{metric_name}'] = value
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def compute_perplexity(self, losses: List[float]) -> float:
        """Compute perplexity from losses."""
        avg_loss = np.mean(losses)
        return np.exp(avg_loss)
    
    def compute_custom_metric(self, predictions: np.ndarray, 
                            labels: np.ndarray,
                            metric_fn: callable) -> float:
        """Compute a custom metric using a provided function."""
        return metric_fn(predictions, labels)