# Contributing to Transformers Fine-Tuning Project

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

### Formatting

We use `black` for code formatting and `isort` for import sorting:

```bash
black .
isort .
```

### Linting

Use `flake8` for linting:

```bash
flake8 .
```

## Project Structure

When adding new features, follow the existing structure:

- `config/` - Configuration files and managers
- `data/` - Data loading and preprocessing
- `models/` - Model architectures and factories
- `training/` - Training loops and utilities
- `evaluation/` - Evaluation and metrics
- `utils/` - General utilities
- `experiments/` - Example experiments
- `tests/` - Unit tests

## Adding New Features

### Adding a New Model Type

1. Update `models/model_factory.py`:
   ```python
   def _create_your_model_type(self):
       # Implementation
   ```

2. Add configuration template in `config/`

3. Update data loader if needed

4. Add tests in `tests/`

### Adding New Metrics

1. Update `evaluation/metrics.py`:
   ```python
   def _compute_your_metric(self, predictions, labels):
       # Implementation
   ```

2. Add metric to configuration schema

3. Add tests

### Adding New Preprocessing

1. Update `data/preprocessor.py`:
   ```python
   def your_preprocessing_function(self, text: str) -> str:
       # Implementation
   ```

2. Add configuration option

3. Add tests

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_data_loader.py

# Run with coverage
python -m pytest tests/ --cov=.
```

### Writing Tests

- Write unit tests for all new functionality
- Use descriptive test names
- Test edge cases and error conditions
- Use fixtures for common setup

Example:

```python
import unittest

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        # Setup code
        pass
    
    def test_basic_functionality(self):
        # Test code
        self.assertEqual(result, expected)
    
    def test_edge_case(self):
        # Edge case test
        pass
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Brief description of function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

### README and Guides

- Update README.md for major features
- Add examples to GETTING_STARTED.md
- Create notebooks for complex workflows

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, atomic commits

3. Add tests for new functionality

4. Update documentation as needed

5. Run tests and ensure they pass:
   ```bash
   make test
   ```

6. Format code:
   ```bash
   make format
   ```

7. Create pull request with clear description:
   - What does this PR do?
   - Why is this change needed?
   - Any breaking changes?
   - How to test?

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commits are clear and atomic
- [ ] PR description is complete

## Commit Messages

Use clear, descriptive commit messages:

```
Add support for custom preprocessing functions

- Implement TextPreprocessor class
- Add configuration options
- Add unit tests
- Update documentation
```

Format: `<type>: <subject>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

## Bug Reports

When reporting bugs, include:

1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details:
   - Python version
   - PyTorch version
   - Transformers version
   - OS

## Feature Requests

When requesting features, include:

1. Clear description of the feature
2. Use case and motivation
3. Proposed implementation (if any)
4. Alternatives considered

## Code Review

### For Reviewers

- Be constructive and respectful
- Explain your suggestions
- Distinguish between required changes and suggestions
- Approve when ready

### For Contributors

- Respond to feedback promptly
- Ask questions if unclear
- Make requested changes
- Mark conversations as resolved

## Community Guidelines

- Be respectful and inclusive
- Help others when possible
- Give credit where due
- Focus on constructive feedback
- Follow the code of conduct

## Getting Help

- Check documentation first
- Search existing issues
- Ask in discussions
- Provide context and details

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish package

Thank you for contributing! 🎉