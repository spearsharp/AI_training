.PHONY: help install setup test clean train-bert train-gpt2 train-t5 evaluate

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make setup         - Setup project (create dirs, install deps)"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean generated files"
	@echo "  make train-bert    - Train BERT classification model"
	@echo "  make train-gpt2    - Train GPT-2 generation model"
	@echo "  make train-t5      - Train T5 seq2seq model"
	@echo "  make evaluate      - Evaluate trained model"

install:
	pip install -r requirements.txt

setup: install
	mkdir -p checkpoints logs cache data/raw data/processed results
	@echo "Project setup complete!"

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	@echo "Cleanup complete!"

train-bert:
	python main.py --config config/bert_classification.yaml

train-gpt2:
	python main.py --config config/gpt2_generation.yaml

train-t5:
	python main.py --config config/t5_seq2seq.yaml

evaluate:
	python evaluation/evaluate.py \
		--model_path checkpoints/best_model \
		--config config/bert_classification.yaml \
		--output_path results/evaluation_results.json

format:
	black .
	isort .

lint:
	flake8 .
	mypy .