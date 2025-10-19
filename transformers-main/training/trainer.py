"""
Main trainer class for transformer fine-tuning.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AdamW,
    Adafactor
)
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import logging

from .early_stopping import EarlyStopping
from .checkpoint_manager import CheckpointManager
from ..evaluation.metrics import MetricsCalculator


logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for transformer fine-tuning."""
    
    def __init__(self, model, config, accelerator: Accelerator, logger: logging.Logger):
        self.model = model
        self.config = config
        self.accelerator = accelerator
        self.logger = logger
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        self.early_stopping = self._create_early_stopping()
        self.checkpoint_manager = CheckpointManager(config.training.output_dir)
        self.metrics_calculator = MetricsCalculator(config)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf') if config.training.greater_is_better else float('inf')
        
        # Logging setup
        self.setup_logging()
        
        # Create output directory
        os.makedirs(config.training.output_dir, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging and tracking."""
        self.tensorboard_writer = None
        
        # Setup TensorBoard
        if hasattr(self.config.logging, 'tracker') and 'tensorboard' in self.config.logging.tracker:
            log_dir = os.path.join(self.config.training.output_dir, 'logs')
            self.tensorboard_writer = SummaryWriter(log_dir)
        
        # Setup Weights & Biases
        if hasattr(self.config.logging, 'tracker') and 'wandb' in self.config.logging.tracker:
            if self.accelerator.is_main_process:
                wandb.init(
                    project=self.config.logging.project_name,
                    name=self.config.logging.experiment_name,
                    config=dict(self.config)
                )
    
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        optimizer_config = self.config.training.optimizer
        
        if optimizer_config.type.lower() == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=optimizer_config.lr,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay
            )
        elif optimizer_config.type.lower() == 'adafactor':
            return Adafactor(
                self.model.parameters(),
                lr=optimizer_config.lr,
                scale_parameter=getattr(optimizer_config, 'scale_parameter', True),
                relative_step_size=getattr(optimizer_config, 'relative_step_size', True),
                warmup_init=getattr(optimizer_config, 'warmup_init', False)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.type}")
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        scheduler_config = self.config.training.scheduler
        
        if scheduler_config.type.lower() == 'linear':
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=scheduler_config.num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_config.type.lower() == 'cosine':
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=scheduler_config.num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config.type}")
    
    def _create_early_stopping(self):
        """Create early stopping callback."""
        es_config = self.config.training.early_stopping
        return EarlyStopping(
            patience=es_config.patience,
            min_delta=es_config.min_delta,
            mode=es_config.mode
        )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Calculate total steps
        num_training_steps = len(train_loader) * self.config.training.num_epochs
        num_training_steps //= self.config.training.gradient_accumulation_steps
        
        # Create scheduler
        self.scheduler = self._create_scheduler(num_training_steps)
        
        # Prepare for distributed training
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )
        
        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Train one epoch
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            if epoch % 1 == 0 or epoch == self.config.training.num_epochs - 1:
                val_metrics = self._validate_epoch(val_loader)
                
                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # Check for improvement
                current_metric = val_metrics[self.config.training.metric_for_best_model]
                if self._is_improvement(current_metric):
                    self.best_metric = current_metric
                    self._save_best_model()
                
                # Early stopping check
                if self.early_stopping(val_metrics[self.config.training.early_stopping.metric]):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % getattr(self.config.training, 'save_epochs', 1) == 0:
                self._save_checkpoint(epoch)
        
        self.logger.info("Training completed!")
        
        # Load best model if specified
        if self.config.training.load_best_model_at_end:
            self._load_best_model()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.training.gradient_accumulation_steps
            
            # Backward pass
            self.accelerator.backward(loss)
            
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            
            # Update weights
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.training.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.training.learning_rate
                    self._log_training_step(loss.item() * self.config.training.gradient_accumulation_steps, current_lr)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.training.gradient_accumulation_steps:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return {'train_loss': total_loss / num_batches}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not self.accelerator.is_local_main_process):
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Collect predictions and labels for metrics
                predictions = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else None
                if predictions is not None:
                    all_predictions.extend(self.accelerator.gather(predictions).cpu().numpy())
                    all_labels.extend(self.accelerator.gather(batch['labels']).cpu().numpy())
        
        # Calculate metrics
        metrics = {'eval_loss': total_loss / len(val_loader)}
        
        if all_predictions and all_labels:
            computed_metrics = self.metrics_calculator.compute_metrics(all_predictions, all_labels)
            metrics.update(computed_metrics)
        
        return metrics
    
    def _is_improvement(self, current_metric: float) -> bool:
        """Check if current metric is an improvement."""
        if self.config.training.greater_is_better:
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    
    def _save_best_model(self):
        """Save the best model."""
        if self.accelerator.is_main_process:
            best_model_path = os.path.join(self.config.training.output_dir, 'best_model')
            self.accelerator.save_model(self.model, best_model_path)
            self.logger.info(f"Saved best model to {best_model_path}")
    
    def _load_best_model(self):
        """Load the best model."""
        best_model_path = os.path.join(self.config.training.output_dir, 'best_model')
        if os.path.exists(best_model_path):
            self.accelerator.load_model(self.model, best_model_path)
            self.logger.info(f"Loaded best model from {best_model_path}")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.accelerator.get_state_dict(self.model),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_metric': self.best_metric,
                'config': dict(self.config)
            }
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(checkpoint, epoch)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int):
        """Log training and validation metrics."""
        # Console logging
        self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['train_loss']:.4f}")
        for metric, value in val_metrics.items():
            self.logger.info(f"Epoch {epoch + 1} - {metric}: {value:.4f}")
        
        # TensorBoard logging
        if self.tensorboard_writer:
            for metric, value in train_metrics.items():
                self.tensorboard_writer.add_scalar(f"train/{metric}", value, epoch)
            for metric, value in val_metrics.items():
                self.tensorboard_writer.add_scalar(f"val/{metric}", value, epoch)
        
        # Weights & Biases logging
        if hasattr(self.config.logging, 'tracker') and 'wandb' in self.config.logging.tracker:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict['epoch'] = epoch
            wandb.log(log_dict)
    
    def _log_training_step(self, loss: float, learning_rate: float):
        """Log training step metrics."""
        # TensorBoard logging
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('train/step_loss', loss, self.global_step)
            self.tensorboard_writer.add_scalar('train/learning_rate', learning_rate, self.global_step)
        
        # Weights & Biases logging
        if hasattr(self.config.logging, 'tracker') and 'wandb' in self.config.logging.tracker:
            wandb.log({
                'train/step_loss': loss,
                'train/learning_rate': learning_rate,
                'global_step': self.global_step
            })