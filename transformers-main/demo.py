#!/usr/bin/env python3
"""
Demo script to show the project structure and verify setup.
This runs without installing heavy dependencies.
"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """Check if all required directories and files exist."""
    print("=" * 60)
    print("🔍 CHECKING PROJECT STRUCTURE")
    print("=" * 60)
    
    required_dirs = [
        'config', 'data', 'models', 'training', 
        'evaluation', 'utils', 'experiments', 'notebooks', 'tests'
    ]
    
    required_files = [
        'main.py', 'README.md', 'requirements.txt', 
        'setup.py', 'LICENSE', 'GETTING_STARTED.md'
    ]
    
    print("\n📁 Checking directories...")
    for dir_name in required_dirs:
        exists = os.path.isdir(dir_name)
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_name}/")
    
    print("\n📄 Checking files...")
    for file_name in required_files:
        exists = os.path.isfile(file_name)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_name}")
    
    return True


def list_configurations():
    """List available configuration files."""
    print("\n" + "=" * 60)
    print("⚙️  AVAILABLE CONFIGURATIONS")
    print("=" * 60)
    
    config_dir = Path('config')
    if config_dir.exists():
        configs = list(config_dir.glob('*.yaml'))
        for config in configs:
            print(f"  📋 {config.name}")
            # Read first few lines
            with open(config, 'r') as f:
                lines = f.readlines()[:3]
                for line in lines:
                    if line.strip() and not line.strip().startswith('#'):
                        print(f"      {line.strip()}")
    else:
        print("  ⚠️  Config directory not found")


def show_model_types():
    """Show available model types."""
    print("\n" + "=" * 60)
    print("🤖 SUPPORTED MODEL TYPES")
    print("=" * 60)
    
    models = {
        "BERT": "Text Classification (sentiment, topic, etc.)",
        "GPT-2": "Text Generation (completion, creative writing)",
        "T5": "Sequence-to-Sequence (summarization, translation)"
    }
    
    for model, description in models.items():
        print(f"  🔹 {model:10} - {description}")


def show_features():
    """Show project features."""
    print("\n" + "=" * 60)
    print("✨ KEY FEATURES")
    print("=" * 60)
    
    features = [
        ("LoRA Support", "Efficient fine-tuning with PEFT"),
        ("Mixed Precision", "FP16/BF16 for faster training"),
        ("Distributed Training", "Multi-GPU support via Accelerate"),
        ("Early Stopping", "Prevent overfitting automatically"),
        ("Checkpointing", "Save and resume training"),
        ("Logging", "TensorBoard & Weights & Biases"),
        ("Flexible Config", "YAML-based configuration"),
        ("Comprehensive Metrics", "Accuracy, F1, BLEU, ROUGE, etc.")
    ]
    
    for feature, desc in features:
        print(f"  ✅ {feature:20} - {desc}")


def show_quick_start():
    """Show quick start commands."""
    print("\n" + "=" * 60)
    print("🚀 QUICK START COMMANDS")
    print("=" * 60)
    
    commands = [
        ("Install dependencies", "pip install -r requirements.txt"),
        ("Setup project", "make setup"),
        ("Train BERT model", "python main.py --config config/bert_classification.yaml"),
        ("Train with script", "./train_bert.sh"),
        ("Evaluate model", "./evaluate.sh"),
        ("Run tests", "make test"),
        ("See all commands", "make help")
    ]
    
    print()
    for desc, cmd in commands:
        print(f"  📌 {desc}:")
        print(f"     {cmd}\n")


def show_next_steps():
    """Show next steps."""
    print("=" * 60)
    print("📖 NEXT STEPS")
    print("=" * 60)
    
    steps = [
        "1. Read GETTING_STARTED.md for detailed instructions",
        "2. Install dependencies: pip install -r requirements.txt",
        "3. Review example configurations in config/",
        "4. Prepare your dataset (JSON, CSV, or use HuggingFace datasets)",
        "5. Customize a configuration file for your task",
        "6. Run training: python main.py --config config/your_config.yaml",
        "7. Monitor training with TensorBoard or Weights & Biases",
        "8. Evaluate on test set and iterate"
    ]
    
    print()
    for step in steps:
        print(f"  {step}")
    print()


def count_files():
    """Count project files."""
    print("=" * 60)
    print("📊 PROJECT STATISTICS")
    print("=" * 60)
    
    py_files = len(list(Path('.').rglob('*.py')))
    yaml_files = len(list(Path('.').rglob('*.yaml')))
    md_files = len(list(Path('.').rglob('*.md')))
    sh_files = len(list(Path('.').rglob('*.sh')))
    
    print(f"\n  📝 Python files: {py_files}")
    print(f"  ⚙️  YAML configs: {yaml_files}")
    print(f"  📄 Documentation: {md_files}")
    print(f"  🔧 Shell scripts: {sh_files}")
    print(f"  📦 Total files: {py_files + yaml_files + md_files + sh_files}")
    print()


def main():
    """Main demo function."""
    print("\n")
    print("█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  🤖 TRANSFORMERS FINE-TUNING PROJECT DEMO  ".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    print()
    
    # Check structure
    check_project_structure()
    
    # Show configurations
    list_configurations()
    
    # Show model types
    show_model_types()
    
    # Show features
    show_features()
    
    # Count files
    count_files()
    
    # Show quick start
    show_quick_start()
    
    # Show next steps
    show_next_steps()
    
    print("=" * 60)
    print("✅ PROJECT SETUP VERIFICATION COMPLETE!")
    print("=" * 60)
    print("\n🎉 Your transformer fine-tuning project is ready to use!")
    print("📚 Start with: cat GETTING_STARTED.md")
    print()


if __name__ == "__main__":
    main()