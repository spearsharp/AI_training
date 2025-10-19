# AI_training
Applied AI Training with Material Pool Management

## Overview
This repository provides a comprehensive Material Pool system for managing AI training resources. The Material Pool allows you to organize, store, and retrieve various types of materials such as datasets, models, documentation, scripts, and configuration files.

## Features

### Material Pool System
- **Material Management**: Add, retrieve, update, and remove materials
- **Type Organization**: Organize materials by type (dataset, model, documentation, script, config)
- **Search Functionality**: Search materials by name or description
- **Metadata Support**: Store additional metadata with each material
- **JSON Export**: Export the entire pool to JSON format
- **Timestamps**: Automatic tracking of creation and update times

### Material Types
- **DATASET**: Training and testing datasets
- **MODEL**: Machine learning models
- **DOCUMENTATION**: Guides, tutorials, and documentation
- **SCRIPT**: Training and utility scripts
- **CONFIG**: Configuration files and hyperparameters
- **OTHER**: Other miscellaneous materials

## Installation

No additional dependencies required! The material pool uses only Python standard library.

```bash
git clone https://github.com/spearsharp/AI_training.git
cd AI_training
```

## Quick Start

### Basic Usage

```python
from material_pool import MaterialPool, Material, MaterialType

# Create a material pool
pool = MaterialPool()

# Add a dataset
dataset = Material(
    name="MNIST Dataset",
    material_type=MaterialType.DATASET,
    description="Handwritten digits dataset",
    path="/data/mnist",
    metadata={"size": "60000 samples"}
)
pool.add_material(dataset)

# Add a model
model = Material(
    name="CNN Model",
    material_type=MaterialType.MODEL,
    description="Convolutional Neural Network",
    path="/models/cnn.h5",
    metadata={"accuracy": "98.5%"}
)
pool.add_material(model)

# List all materials
for material in pool.list_materials():
    print(f"{material.name}: {material.description}")

# Get materials by type
datasets = pool.get_materials_by_type(MaterialType.DATASET)

# Search materials
results = pool.search_materials("CNN")

# Export to JSON
json_data = pool.export_to_json()
```

### Run Example

See a complete working example:

```bash
python3 example_usage.py
```

## API Reference

### Material Class

```python
Material(
    name: str,                          # Material name
    material_type: MaterialType,        # Type of material
    description: str = "",              # Optional description
    path: str = "",                     # Path to resource
    metadata: Dict[str, Any] = None     # Additional metadata
)
```

**Methods:**
- `update(**kwargs)`: Update material attributes
- `to_dict()`: Convert to dictionary

### MaterialPool Class

```python
MaterialPool()  # Create an empty pool
```

**Methods:**
- `add_material(material: Material) -> bool`: Add a material to the pool
- `get_material(name: str) -> Optional[Material]`: Retrieve a material by name
- `remove_material(name: str) -> bool`: Remove a material from the pool
- `list_materials(material_type: Optional[MaterialType] = None) -> List[Material]`: List all materials (optionally filtered by type)
- `update_material(name: str, **kwargs) -> bool`: Update a material's attributes
- `get_materials_by_type(material_type: MaterialType) -> List[Material]`: Get all materials of a specific type
- `search_materials(query: str) -> List[Material]`: Search materials by name or description
- `count() -> int`: Get total number of materials
- `clear()`: Remove all materials
- `export_to_json() -> str`: Export pool to JSON

## Use Cases

### 1. Dataset Management
Organize and track all your training datasets in one place:
```python
dataset = Material(
    name="ImageNet",
    material_type=MaterialType.DATASET,
    description="Large-scale image dataset",
    path="/data/imagenet",
    metadata={"samples": 1000000, "classes": 1000}
)
pool.add_material(dataset)
```

### 2. Model Versioning
Keep track of different model versions and their performance:
```python
model_v1 = Material(
    name="ResNet50_v1",
    material_type=MaterialType.MODEL,
    description="ResNet50 baseline model",
    path="/models/resnet50_v1.pth",
    metadata={"accuracy": 92.1, "version": "1.0"}
)
pool.add_material(model_v1)
```

### 3. Documentation Organization
Maintain all training-related documentation:
```python
doc = Material(
    name="Training Pipeline",
    material_type=MaterialType.DOCUMENTATION,
    description="End-to-end training pipeline guide",
    path="/docs/pipeline.md",
    metadata={"author": "AI Team"}
)
pool.add_material(doc)
```

### 4. Configuration Management
Store and retrieve hyperparameter configurations:
```python
config = Material(
    name="Production Config",
    material_type=MaterialType.CONFIG,
    description="Production hyperparameters",
    path="/config/prod.json",
    metadata={"lr": 0.001, "batch_size": 64}
)
pool.add_material(config)
```

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues.

## License

This project is open source and available for use in AI training projects.
