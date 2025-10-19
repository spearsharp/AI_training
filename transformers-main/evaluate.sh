#!/bin/bash

# Evaluation script

echo "Starting Model Evaluation..."

python evaluation/evaluate.py \
    --model_path checkpoints/best_model \
    --config config/bert_classification.yaml \
    --data_path data/test.json \
    --output_path results/evaluation_results.json

echo "Evaluation completed!"