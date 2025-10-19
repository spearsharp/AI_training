"""
Example usage of the Material Pool for AI Training Project

This script demonstrates how to use the MaterialPool to manage AI training materials.
"""

from material_pool import MaterialPool, Material, MaterialType


def main():
    """Demonstrate material pool usage."""
    # Create a material pool
    pool = MaterialPool()
    
    print("=" * 60)
    print("Material Pool for AI Training Project - Example Usage")
    print("=" * 60)
    print()
    
    # Add various materials
    print("1. Adding materials to the pool...")
    
    # Add a dataset
    dataset1 = Material(
        name="MNIST Dataset",
        material_type=MaterialType.DATASET,
        description="Handwritten digits dataset for image classification",
        path="/data/mnist",
        metadata={"size": "60000 samples", "format": "images"}
    )
    pool.add_material(dataset1)
    print(f"   Added: {dataset1}")
    
    # Add a model
    model1 = Material(
        name="CNN Model v1",
        material_type=MaterialType.MODEL,
        description="Convolutional Neural Network for digit recognition",
        path="/models/cnn_v1.h5",
        metadata={"accuracy": "98.5%", "framework": "TensorFlow"}
    )
    pool.add_material(model1)
    print(f"   Added: {model1}")
    
    # Add documentation
    doc1 = Material(
        name="Training Guide",
        material_type=MaterialType.DOCUMENTATION,
        description="Step-by-step guide for training the CNN model",
        path="/docs/training_guide.md",
        metadata={"version": "1.0"}
    )
    pool.add_material(doc1)
    print(f"   Added: {doc1}")
    
    # Add a training script
    script1 = Material(
        name="Train Script",
        material_type=MaterialType.SCRIPT,
        description="Python script for model training",
        path="/scripts/train.py",
        metadata={"language": "Python"}
    )
    pool.add_material(script1)
    print(f"   Added: {script1}")
    
    # Add a configuration file
    config1 = Material(
        name="Hyperparameters Config",
        material_type=MaterialType.CONFIG,
        description="Configuration file for model hyperparameters",
        path="/config/hyperparams.json",
        metadata={"learning_rate": 0.001, "batch_size": 32}
    )
    pool.add_material(config1)
    print(f"   Added: {config1}")
    
    print()
    print(f"Total materials in pool: {pool.count()}")
    print()
    
    # List all materials
    print("2. Listing all materials in the pool...")
    for material in pool.list_materials():
        print(f"   - {material.name} ({material.material_type.value})")
    print()
    
    # Get materials by type
    print("3. Getting materials by type...")
    print("   Datasets:")
    for material in pool.get_materials_by_type(MaterialType.DATASET):
        print(f"      - {material.name}: {material.description}")
    
    print("   Models:")
    for material in pool.get_materials_by_type(MaterialType.MODEL):
        print(f"      - {material.name}: {material.description}")
    print()
    
    # Retrieve a specific material
    print("4. Retrieving a specific material...")
    mnist = pool.get_material("MNIST Dataset")
    if mnist:
        print(f"   Found: {mnist.name}")
        print(f"   Type: {mnist.material_type.value}")
        print(f"   Description: {mnist.description}")
        print(f"   Path: {mnist.path}")
        print(f"   Metadata: {mnist.metadata}")
    print()
    
    # Search materials
    print("5. Searching for materials containing 'CNN'...")
    results = pool.search_materials("CNN")
    for material in results:
        print(f"   - {material.name}: {material.description}")
    print()
    
    # Update a material
    print("6. Updating a material...")
    pool.update_material(
        "CNN Model v1",
        description="Improved CNN for digit recognition",
        metadata={"accuracy": "99.1%", "framework": "TensorFlow", "version": "v1.1"}
    )
    updated_model = pool.get_material("CNN Model v1")
    print(f"   Updated: {updated_model.name}")
    print(f"   New description: {updated_model.description}")
    print(f"   New metadata: {updated_model.metadata}")
    print()
    
    # Export to JSON
    print("7. Exporting pool to JSON...")
    json_export = pool.export_to_json()
    print("   Export successful! (First 200 characters)")
    print(f"   {json_export[:200]}...")
    print()
    
    # Remove a material
    print("8. Removing a material...")
    removed = pool.remove_material("Train Script")
    if removed:
        print(f"   Removed 'Train Script'")
        print(f"   Remaining materials: {pool.count()}")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
