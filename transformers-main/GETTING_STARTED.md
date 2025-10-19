# Getting Started with Transformers Fine-Tuning

This guide will help you get started with fine-tuning transformer models using this project.

## Installation

### 1. Clone the repository (if from Git) or navigate to the project directory

```bash
cd /Users/project/python/Ai/transformers-main
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

## Quick Start Examples

### Example 1: BERT Text Classification

Train a BERT model for binary text classification (e.g., sentiment analysis):

```bash
python main.py --config config/bert_classification.yaml
```

Or use the shell script:

```bash
chmod +x train_bert.sh
./train_bert.sh
```

### Example 2: GPT-2 Text Generation

Fine-tune GPT-2 for text generation:

```bash
python main.py --config config/gpt2_generation.yaml
```

### Example 3: T5 Sequence-to-Sequence

Train T5 for sequence-to-sequence tasks:

```bash
python main.py --config config/t5_seq2seq.yaml
```

## Custom Dataset

### Preparing Your Data

Your data should be in JSON Lines format (`.jsonl`) or CSV format:

```json
{"text": "This is a sample text", "label": 0}
{"text": "Another example", "label": 1}
```

### Configuration

Modify the configuration file to point to your data:

```yaml
data:
  dataset_name: "/path/to/your/data.jsonl"
  text_column: "text"
  label_column: "label"
  max_length: 512
```

## Advanced Features

### Using LoRA for Efficient Fine-Tuning

LoRA (Low-Rank Adaptation) allows efficient fine-tuning with fewer trainable parameters:

```yaml
model:
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["query", "value"]
```

### Distributed Training

For multi-GPU training, use the Accelerate library:

```bash
accelerate config  # Configure once
accelerate launch main.py --config config/bert_classification.yaml
```

### Weights & Biases Integration

Enable W&B logging in your config:

```yaml
logging:
  tracker: "wandb"
  project_name: "my-project"
  experiment_name: "bert-experiment-1"
```

Then run with W&B:

```bash
wandb login
python main.py --config config/bert_classification.yaml
```

### Mixed Precision Training

Enable FP16 or BF16 for faster training:

```yaml
training:
  mixed_precision: "fp16"  # or "bf16"
```

## Evaluation

Evaluate your trained model:

```bash
python evaluation/evaluate.py \
    --model_path checkpoints/best_model \
    --config config/bert_classification.yaml \
    --output_path results/eval_results.json
```

## Project Structure Overview

```
transformers-main/
├── config/              # Configuration files
│   ├── bert_classification.yaml
│   ├── gpt2_generation.yaml
│   └── t5_seq2seq.yaml
├── data/               # Data loading and preprocessing
│   ├── data_loader.py
│   └── preprocessor.py
├── models/             # Model creation and utilities
│   └── model_factory.py
├── training/           # Training logic
│   ├── trainer.py
│   ├── early_stopping.py
│   └── checkpoint_manager.py
├── evaluation/         # Evaluation and metrics
│   ├── evaluator.py
│   └── metrics.py
├── utils/             # Utilities
│   ├── logger.py
│   ├── seed.py
│   └── gpu_utils.py
├── experiments/       # Example experiments
└── main.py           # Main training script
```

## Configuration Options

### Key Configuration Sections

1. **Model Configuration**
   - Model name/path
   - Model type (classification, generation, seq2seq)
   - LoRA settings
   - Model-specific parameters

2. **Data Configuration**
   - Dataset path or name
   - Column names
   - Maximum sequence length
   - Preprocessing options

3. **Training Configuration**
   - Batch size
   - Learning rate
   - Number of epochs
   - Optimizer and scheduler settings
   - Gradient accumulation
   - Mixed precision

4. **Evaluation Configuration**
   - Metrics to compute
   - Evaluation frequency

## Common Use Cases

### Sentiment Analysis

```yaml
model:
  name: "bert-base-uncased"
  type: "classification"
  num_classes: 2

data:
  dataset_name: "imdb"
  text_column: "text"
  label_column: "label"
```

### Question Generation

```yaml
model:
  name: "t5-small"
  type: "seq2seq"

data:
  dataset_name: "squad"
  input_column: "context"
  target_column: "question"
  prefix: "generate question: "
```

### Text Completion

```yaml
model:
  name: "gpt2"
  type: "generation"

data:
  dataset_name: "wikitext"
  text_column: "text"
```

## Tips for Better Results

1. **Start Small**: Begin with a small model and dataset to verify everything works
2. **Monitor Training**: Use TensorBoard or W&B to track training progress
3. **Adjust Learning Rate**: Try different learning rates (typically 1e-5 to 5e-5 for BERT)
4. **Use Early Stopping**: Prevent overfitting with patience-based early stopping
5. **Experiment with Batch Size**: Larger batches can stabilize training
6. **Try LoRA**: For large models, LoRA can significantly reduce memory usage

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce batch size
- Enable gradient accumulation
- Use mixed precision (FP16)
- Enable LoRA
- Try gradient checkpointing

### Slow Training

- Increase batch size (if memory allows)
- Use mixed precision
- Increase number of workers for data loading
- Use distributed training for multiple GPUs

### Poor Performance

- Try different learning rates
- Increase training epochs
- Adjust warmup steps
- Check data preprocessing
- Verify label distribution

## Next Steps

1. Explore the example experiments in `experiments/`
2. Modify configurations for your specific task
3. Add custom preprocessing in `data/preprocessor.py`
4. Implement custom metrics in `evaluation/metrics.py`
5. Check out the notebooks in `notebooks/` for interactive examples

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## Support

For issues or questions, please check:
- Project documentation
- Configuration examples
- Test files in `tests/`

Happy fine-tuning! 🚀