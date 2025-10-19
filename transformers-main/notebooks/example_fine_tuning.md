# Transformer Fine-Tuning Example Notebook

This notebook demonstrates how to use the transformers-main project for fine-tuning models.

## Setup

```python
import sys
sys.path.append('..')

from config.config_manager import ConfigManager
from data.data_loader import CustomDataLoader
from models.model_factory import ModelFactory
from utils.logger import setup_logger
from utils.seed import set_seed
```

## Load Configuration

```python
config_manager = ConfigManager('../config/bert_classification.yaml')
config = config_manager.get_config()
print(f"Model: {config.model.name}")
print(f"Task: {config.model.type}")
```

## Set Random Seed

```python
set_seed(42)
```

## Load Data

```python
data_loader = CustomDataLoader(config)
train_dataset, val_dataset, test_dataset = data_loader.load_datasets()

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset) if test_dataset else 'N/A'}")
```

## Create Model

```python
model_factory = ModelFactory(config)
model = model_factory.create_model()

# Get model info
model_info = model_factory.get_model_size(model)
print(f"Parameters: {model_info['parameters']:,}")
print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
print(f"Size: {model_info['size_mb']:.2f} MB")
```

## Quick Training Example

```python
from torch.utils.data import DataLoader
from training.trainer import Trainer
from accelerate import Accelerator

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize accelerator
accelerator = Accelerator()

# Create trainer
logger = setup_logger()
trainer = Trainer(model, config, accelerator, logger)

# Train (this will take time!)
# trainer.train(train_loader, val_loader)
```

## Evaluation Example

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(config, logger)

# Evaluate on validation set
# results = evaluator.evaluate(model, val_loader)
# print(results)
```

## Making Predictions

```python
# Example: Make predictions on new data
sample_text = "This is a sample text for prediction"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.name)

# Tokenize
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    
print(f"Prediction: {predictions.item()}")
```

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot training metrics (example)
# This would use logged metrics from training

epochs = [1, 2, 3, 4, 5]
train_loss = [0.5, 0.4, 0.3, 0.25, 0.2]
val_loss = [0.55, 0.45, 0.35, 0.32, 0.28]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
```

## Next Steps

1. Modify the configuration file for your specific task
2. Prepare your custom dataset
3. Run full training with `python main.py --config config/your_config.yaml`
4. Monitor training with TensorBoard or Weights & Biases
5. Evaluate on test set and iterate