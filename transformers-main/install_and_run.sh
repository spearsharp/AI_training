#!/bin/bash

# Installation and Quick Start Guide for Full Transformer Training

echo "=========================================================================="
echo "🚀 TRANSFORMERS FINE-TUNING - INSTALLATION GUIDE"
echo "=========================================================================="
echo ""

echo "This script will help you install dependencies and run your first real"
echo "transformer model training."
echo ""
echo "⚠️  WARNING: This will download ~2-3 GB of packages"
echo ""
read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Installation cancelled."
    exit 1
fi

echo ""
echo "=========================================================================="
echo "📦 STEP 1: Installing Core Dependencies"
echo "=========================================================================="
echo ""

# Install PyTorch (CPU version - smaller download)
echo "Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing Transformers and related packages..."
pip install transformers datasets tokenizers

echo ""
echo "Installing training utilities..."
pip install accelerate tensorboard

echo ""
echo "Installing evaluation packages..."
pip install scikit-learn

echo ""
echo "Installing configuration and logging..."
pip install pyyaml omegaconf rich

echo ""
echo "=========================================================================="
echo "✅ STEP 2: Installation Complete!"
echo "=========================================================================="
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}')"

echo ""
echo "=========================================================================="
echo "🎯 STEP 3: What You Can Do Now"
echo "=========================================================================="
echo ""

echo "1️⃣  View example configurations:"
echo "   ls -la config/"
echo ""

echo "2️⃣  Train BERT for text classification:"
echo "   python main.py --config config/bert_classification.yaml"
echo ""

echo "3️⃣  Train GPT-2 for text generation:"
echo "   python main.py --config config/gpt2_generation.yaml"
echo ""

echo "4️⃣  Use convenience scripts:"
echo "   ./train_bert.sh"
echo "   ./train_gpt2.sh"
echo ""

echo "5️⃣  Run with custom data:"
echo "   - Edit config/*.yaml to point to your data"
echo "   - Format: JSON Lines or CSV"
echo "   - Then run: python main.py --config config/your_config.yaml"
echo ""

echo "=========================================================================="
echo "📚 STEP 4: Next Steps"
echo "=========================================================================="
echo ""

echo "✅ Read the getting started guide:"
echo "   cat GETTING_STARTED.md"
echo ""

echo "✅ Review example configurations:"
echo "   cat config/bert_classification.yaml"
echo ""

echo "✅ Check the project structure:"
echo "   python demo.py"
echo ""

echo "✅ When ready, start training:"
echo "   python main.py --config config/bert_classification.yaml"
echo ""

echo "=========================================================================="
echo "💡 TIPS FOR SUCCESS"
echo "=========================================================================="
echo ""

echo "• Start with small datasets to test"
echo "• Monitor training with: tensorboard --logdir checkpoints/"
echo "• Use smaller models first (bert-base instead of bert-large)"
echo "• Enable LoRA in config for efficient fine-tuning"
echo "• Adjust batch_size based on your RAM/GPU memory"
echo ""

echo "=========================================================================="
echo "🎉 Setup Complete! Happy Fine-Tuning!"
echo "=========================================================================="
echo ""