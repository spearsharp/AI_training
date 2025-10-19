"""
Data loading utilities for transformer fine-tuning.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import logging

from .preprocessor import TextPreprocessor


logger = logging.getLogger(__name__)


class CustomDataLoader:
    """Custom data loader for various NLP tasks."""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            cache_dir=config.data.cache_dir
        )
        self.preprocessor = TextPreprocessor(config.data.preprocessing)
        
        # Add special tokens if needed
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Setup tokenizer with special tokens."""
        if self.config.model.type == "generation":
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.config.model.type == "seq2seq":
            # T5 already has proper tokens, but we can customize if needed
            pass
    
    def load_datasets(self) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
        """Load train, validation, and test datasets."""
        logger.info(f"Loading dataset: {self.config.data.dataset_name}")
        
        # Load from HuggingFace Hub or local files
        if self._is_local_dataset():
            datasets = self._load_local_dataset()
        else:
            datasets = self._load_huggingface_dataset()
        
        # Preprocess datasets
        train_dataset = self._preprocess_dataset(datasets['train'], is_train=True)
        val_dataset = self._preprocess_dataset(datasets['validation'], is_train=False)
        
        test_dataset = None
        if 'test' in datasets and datasets['test'] is not None:
            test_dataset = self._preprocess_dataset(datasets['test'], is_train=False)
        
        logger.info(f"Loaded datasets - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset) if test_dataset else 'None'}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _is_local_dataset(self) -> bool:
        """Check if dataset is local file path."""
        return os.path.exists(self.config.data.dataset_name)
    
    def _load_local_dataset(self) -> DatasetDict:
        """Load dataset from local files."""
        data_path = Path(self.config.data.dataset_name)
        
        if data_path.is_file():
            # Single file - split it
            if data_path.suffix == '.json':
                data = pd.read_json(data_path, lines=True)
            elif data_path.suffix == '.csv':
                data = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
            # Split data
            train_data, val_data, test_data = self._split_dataset(data)
            
        else:
            # Directory with separate files
            train_data = self._load_file(data_path / "train.json")
            val_data = self._load_file(data_path / "validation.json")
            test_data = self._load_file(data_path / "test.json")
        
        return DatasetDict({
            'train': Dataset.from_pandas(train_data),
            'validation': Dataset.from_pandas(val_data),
            'test': Dataset.from_pandas(test_data) if test_data is not None else None
        })
    
    def _load_huggingface_dataset(self) -> DatasetDict:
        """Load dataset from HuggingFace Hub."""
        dataset_config = getattr(self.config.data, 'dataset_config', None)
        
        dataset = load_dataset(
            self.config.data.dataset_name,
            dataset_config,
            cache_dir=self.config.data.cache_dir
        )
        
        # Handle different split configurations
        train_split = self.config.data.train_split
        val_split = self.config.data.validation_split
        test_split = getattr(self.config.data, 'test_split', None)
        
        datasets = {}
        datasets['train'] = dataset[train_split] if train_split in dataset else None
        
        # Handle validation split (might be a slice of another split)
        if ':' in val_split or '[' in val_split:
            # It's a slice like "test[:10%]"
            base_split = val_split.split('[')[0]
            datasets['validation'] = dataset[val_split]
        else:
            datasets['validation'] = dataset[val_split] if val_split in dataset else None
        
        # Handle test split
        if test_split:
            datasets['test'] = dataset[test_split] if test_split in dataset else None
        else:
            datasets['test'] = None
        
        return DatasetDict(datasets)
    
    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single file."""
        if not file_path.exists():
            return None
        
        if file_path.suffix == '.json':
            return pd.read_json(file_path, lines=True)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _split_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Split dataset into train/val/test."""
        # Simple 80/10/10 split
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:] if len(data) > train_size + val_size else None
        
        return train_data, val_data, test_data
    
    def _preprocess_dataset(self, dataset: Dataset, is_train: bool = True) -> Dataset:
        """Preprocess dataset for the specific task."""
        if self.config.model.type == "classification":
            return self._preprocess_classification(dataset, is_train)
        elif self.config.model.type == "generation":
            return self._preprocess_generation(dataset, is_train)
        elif self.config.model.type == "seq2seq":
            return self._preprocess_seq2seq(dataset, is_train)
        else:
            raise ValueError(f"Unknown model type: {self.config.model.type}")
    
    def _preprocess_classification(self, dataset: Dataset, is_train: bool) -> Dataset:
        """Preprocess data for classification tasks."""
        
        def tokenize_function(examples):
            # Get text column
            texts = examples[self.config.data.text_column]
            
            # Preprocess texts
            if hasattr(self.preprocessor, 'preprocess'):
                texts = [self.preprocessor.preprocess(text) for text in texts]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.config.data.max_length,
                return_tensors='pt'
            )
            
            # Add labels
            if self.config.data.label_column in examples:
                tokenized['labels'] = examples[self.config.data.label_column]
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _preprocess_generation(self, dataset: Dataset, is_train: bool) -> Dataset:
        """Preprocess data for text generation tasks."""
        
        def tokenize_function(examples):
            texts = examples[self.config.data.text_column]
            
            # Preprocess texts
            if hasattr(self.preprocessor, 'preprocess'):
                texts = [self.preprocessor.preprocess(text) for text in texts]
            
            # For generation, input and labels are the same (causal LM)
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.config.data.max_length,
                return_tensors='pt'
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _preprocess_seq2seq(self, dataset: Dataset, is_train: bool) -> Dataset:
        """Preprocess data for sequence-to-sequence tasks."""
        
        def tokenize_function(examples):
            inputs = examples[self.config.data.input_column]
            targets = examples[self.config.data.target_column]
            
            # Add prefix if specified
            if hasattr(self.config.data, 'prefix') and self.config.data.prefix:
                inputs = [self.config.data.prefix + inp for inp in inputs]
            
            # Preprocess
            if hasattr(self.preprocessor, 'preprocess'):
                inputs = [self.preprocessor.preprocess(text) for text in inputs]
                targets = [self.preprocessor.preprocess(text) for text in targets]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                truncation=True,
                padding='max_length',
                max_length=self.config.model.max_input_length,
                return_tensors='pt'
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets,
                truncation=True,
                padding='max_length',
                max_length=self.config.model.max_target_length,
                return_tensors='pt'
            )
            
            model_inputs['labels'] = labels['input_ids']
            
            return model_inputs
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )