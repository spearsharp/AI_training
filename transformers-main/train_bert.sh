#!/bin/bash

# Training script for BERT classification

echo "Starting BERT Classification Training..."

python main.py \
    --config config/bert_classification.yaml \
    --debug

echo "Training completed!"