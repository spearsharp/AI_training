#!/bin/bash

# Training script for GPT-2 generation

echo "Starting GPT-2 Generation Training..."

python main.py \
    --config config/gpt2_generation.yaml

echo "Training completed!"