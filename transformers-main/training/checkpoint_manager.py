"""
Checkpoint management for training.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints during training."""
    
    def __init__(self, output_dir: str, max_checkpoints: int = 3):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], epoch: int) -> str:
        """
        Save a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            epoch: Current epoch
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint-epoch-{epoch}"
        checkpoint_path = self.output_dir / checkpoint_name
        
        # Save checkpoint
        checkpoint_file = checkpoint_path / "pytorch_model.bin"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_file)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'global_step': checkpoint.get('global_step', 0),
            'best_metric': checkpoint.get('best_metric', None),
            'timestamp': torch.save.__doc__  # Simple timestamp
        }
        
        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Add to checkpoint list
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'timestamp': metadata['timestamp']
        })
        
        # Sort by epoch
        self.checkpoints.sort(key=lambda x: x['epoch'])
        
        # Remove old checkpoints if necessary
        self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint_file = Path(checkpoint_path) / "pytorch_model.bin"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['epoch'])
        return latest['path']
    
    def get_best_checkpoint(self, metric_key: str = 'best_metric') -> Optional[str]:
        """
        Get path to the best checkpoint based on a metric.
        
        Args:
            metric_key: Key to use for finding best checkpoint
            
        Returns:
            Path to best checkpoint or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        
        best_checkpoint = None
        best_metric = float('-inf')
        
        for checkpoint_info in self.checkpoints:
            try:
                metadata_path = Path(checkpoint_info['path']) / "metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if metric_key in metadata and metadata[metric_key] is not None:
                    if metadata[metric_key] > best_metric:
                        best_metric = metadata[metric_key]
                        best_checkpoint = checkpoint_info['path']
            except Exception as e:
                logger.warning(f"Could not read metadata for {checkpoint_info['path']}: {e}")
        
        return best_checkpoint
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        return self.checkpoints.copy()
    
    def delete_checkpoint(self, checkpoint_path: str):
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            
            # Remove from list
            self.checkpoints = [
                cp for cp in self.checkpoints 
                if cp['path'] != str(checkpoint_path)
            ]
            
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
    
    def _load_existing_checkpoints(self):
        """Load information about existing checkpoints."""
        if not self.output_dir.exists():
            return
        
        for checkpoint_dir in self.output_dir.iterdir():
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint-'):
                metadata_path = checkpoint_dir / "metadata.json"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        self.checkpoints.append({
                            'path': str(checkpoint_dir),
                            'epoch': metadata['epoch'],
                            'timestamp': metadata.get('timestamp', '')
                        })
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {checkpoint_dir}: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch and remove oldest
        self.checkpoints.sort(key=lambda x: x['epoch'])
        
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            self.delete_checkpoint(oldest['path'])
    
    def cleanup_all_checkpoints(self):
        """Delete all checkpoints."""
        for checkpoint_info in self.checkpoints.copy():
            self.delete_checkpoint(checkpoint_info['path'])
        
        self.checkpoints = []
        logger.info("Deleted all checkpoints")