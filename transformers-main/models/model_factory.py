"""
Model factory for creating different transformer models.
"""

import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and configuring transformer models."""
    
    def __init__(self, config):
        self.config = config
        
    def create_model(self):
        """Create model based on configuration."""
        model_type = self.config.model.type.lower()
        
        if model_type == "classification":
            return self._create_classification_model()
        elif model_type == "generation":
            return self._create_generation_model()
        elif model_type == "seq2seq":
            return self._create_seq2seq_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_classification_model(self):
        """Create a sequence classification model."""
        logger.info(f"Creating classification model: {self.config.model.name}")
        
        # Load configuration
        config = AutoConfig.from_pretrained(
            self.config.model.name,
            num_labels=self.config.model.num_classes,
            hidden_dropout_prob=self.config.model.dropout,
            cache_dir=getattr(self.config.data, 'cache_dir', None)
        )
        
        # Create model
        if self.config.model.pretrained:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.name,
                config=config,
                cache_dir=getattr(self.config.data, 'cache_dir', None)
            )
        else:
            model = AutoModelForSequenceClassification.from_config(config)
        
        # Apply LoRA if configured
        if self.config.model.get('use_lora', False):
            model = self._apply_lora(model, TaskType.SEQ_CLS)
        
        return model
    
    def _create_generation_model(self):
        """Create a causal language model for text generation."""
        logger.info(f"Creating generation model: {self.config.model.name}")
        
        # Create model
        if self.config.model.pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                cache_dir=getattr(self.config.data, 'cache_dir', None),
                torch_dtype=torch.float16 if self.config.training.mixed_precision == "fp16" else torch.float32
            )
        else:
            config = AutoConfig.from_pretrained(self.config.model.name)
            model = AutoModelForCausalLM.from_config(config)
        
        # Apply LoRA if configured
        if self.config.model.get('use_lora', False):
            model = self._apply_lora(model, TaskType.CAUSAL_LM)
        
        return model
    
    def _create_seq2seq_model(self):
        """Create a sequence-to-sequence model."""
        logger.info(f"Creating seq2seq model: {self.config.model.name}")
        
        # Create model
        if self.config.model.pretrained:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model.name,
                cache_dir=getattr(self.config.data, 'cache_dir', None),
                torch_dtype=torch.float16 if self.config.training.mixed_precision == "fp16" else torch.float32
            )
        else:
            config = AutoConfig.from_pretrained(self.config.model.name)
            model = AutoModelForSeq2SeqLM.from_config(config)
        
        # Apply LoRA if configured
        if self.config.model.get('use_lora', False):
            model = self._apply_lora(model, TaskType.SEQ_2_SEQ_LM)
        
        return model
    
    def _apply_lora(self, model, task_type):
        """Apply LoRA (Low-Rank Adaptation) to the model."""
        logger.info("Applying LoRA configuration")
        
        lora_config_dict = self.config.model.lora_config
        
        lora_config = LoraConfig(
            task_type=task_type,
            r=lora_config_dict.r,
            lora_alpha=lora_config_dict.lora_alpha,
            lora_dropout=lora_config_dict.lora_dropout,
            target_modules=lora_config_dict.target_modules,
            bias="none",
            inference_mode=False,
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def get_model_size(self, model):
        """Get model size information."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'size_mb': size_all_mb
        }


class ModelWrapper:
    """Wrapper class for models with additional utilities."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def generate(self, input_ids, **kwargs):
        """Generate text (for generation models)."""
        if hasattr(self.model, 'generate'):
            generation_config = self.config.model.get('generation', {})
            generation_kwargs = {**generation_config, **kwargs}
            return self.model.generate(input_ids, **generation_kwargs)
        else:
            raise ValueError("Model does not support text generation")
    
    def save_pretrained(self, save_directory):
        """Save model and configuration."""
        self.model.save_pretrained(save_directory)
    
    def load_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def get_embeddings(self, input_ids):
        """Get embeddings from the model."""
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()(input_ids)
        else:
            raise ValueError("Model does not support embedding extraction")