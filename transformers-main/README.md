# Transformers Fine-Tuning Project

A comprehensive PyTorch-based project for fine-tuning transformer models on custom datasets.

## Features

- Support for multiple transformer architectures (BERT, GPT-2, T5, etc.)
- Configurable training pipeline
- Multi-task learning support
- Distributed training capabilities
- Comprehensive evaluation metrics
- Data preprocessing utilities
- Model checkpointing and resuming
- Tensorboard logging
- Hyperparameter optimization

## Project Structure

```
transformers-main/
├── config/                 # Configuration files
├── data/                  # Data processing utilities
├── models/                # Model architectures and utilities
├── training/              # Training scripts and utilities
├── evaluation/            # Evaluation and metrics
├── utils/                 # General utilities
├── experiments/           # Experiment configurations
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── main.py               # Main training script
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
```bash
python data/prepare_data.py --config config/data_config.yaml
```

3. Train a model:
```bash
python main.py --config config/bert_classification.yaml
```

4. Evaluate:
```bash
python evaluation/evaluate.py --model_path checkpoints/best_model.pt --data_path data/test.json
```

## Configuration

All experiments are configured via YAML files in the `config/` directory. Key configuration options:

- Model architecture and parameters
- Training hyperparameters
- Data preprocessing settings
- Evaluation metrics
- Hardware settings (GPU, distributed training)

## Supported Tasks

- Text Classification
- Named Entity Recognition (NER)
- Question Answering
- Text Generation
- Sequence-to-Sequence tasks
- Multi-label Classification

## Hardware Requirements

- GPU with at least 8GB VRAM recommended
- 16GB+ RAM for large datasets
- CUDA support for optimal performance

## License

MIT License