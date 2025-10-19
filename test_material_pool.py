"""
Unit tests for the Material Pool

Run with: python3 test_material_pool.py
"""

import unittest
from material_pool import MaterialPool, Material, MaterialType


class TestMaterial(unittest.TestCase):
    """Tests for the Material class."""
    
    def test_material_creation(self):
        """Test creating a material."""
        material = Material(
            name="Test Dataset",
            material_type=MaterialType.DATASET,
            description="A test dataset",
            path="/data/test"
        )
        self.assertEqual(material.name, "Test Dataset")
        self.assertEqual(material.material_type, MaterialType.DATASET)
        self.assertEqual(material.description, "A test dataset")
        self.assertEqual(material.path, "/data/test")
    
    def test_material_update(self):
        """Test updating a material."""
        material = Material(
            name="Test Model",
            material_type=MaterialType.MODEL
        )
        material.update(description="Updated description", path="/new/path")
        self.assertEqual(material.description, "Updated description")
        self.assertEqual(material.path, "/new/path")
    
    def test_material_to_dict(self):
        """Test converting material to dictionary."""
        material = Material(
            name="Test",
            material_type=MaterialType.SCRIPT,
            metadata={"key": "value"}
        )
        material_dict = material.to_dict()
        self.assertEqual(material_dict["name"], "Test")
        self.assertEqual(material_dict["material_type"], "script")
        self.assertIn("created_at", material_dict)
        self.assertIn("updated_at", material_dict)


class TestMaterialPool(unittest.TestCase):
    """Tests for the MaterialPool class."""
    
    def setUp(self):
        """Set up a fresh material pool for each test."""
        self.pool = MaterialPool()
    
    def test_add_material(self):
        """Test adding a material to the pool."""
        material = Material("Test", MaterialType.DATASET)
        result = self.pool.add_material(material)
        self.assertTrue(result)
        self.assertEqual(self.pool.count(), 1)
    
    def test_add_duplicate_material(self):
        """Test adding a material with duplicate name."""
        material1 = Material("Test", MaterialType.DATASET)
        material2 = Material("Test", MaterialType.MODEL)
        self.pool.add_material(material1)
        result = self.pool.add_material(material2)
        self.assertFalse(result)
        self.assertEqual(self.pool.count(), 1)
    
    def test_get_material(self):
        """Test retrieving a material."""
        material = Material("Test", MaterialType.DATASET)
        self.pool.add_material(material)
        retrieved = self.pool.get_material("Test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test")
    
    def test_get_nonexistent_material(self):
        """Test retrieving a nonexistent material."""
        retrieved = self.pool.get_material("Nonexistent")
        self.assertIsNone(retrieved)
    
    def test_remove_material(self):
        """Test removing a material."""
        material = Material("Test", MaterialType.DATASET)
        self.pool.add_material(material)
        result = self.pool.remove_material("Test")
        self.assertTrue(result)
        self.assertEqual(self.pool.count(), 0)
    
    def test_remove_nonexistent_material(self):
        """Test removing a nonexistent material."""
        result = self.pool.remove_material("Nonexistent")
        self.assertFalse(result)
    
    def test_list_materials(self):
        """Test listing all materials."""
        material1 = Material("Test1", MaterialType.DATASET)
        material2 = Material("Test2", MaterialType.MODEL)
        self.pool.add_material(material1)
        self.pool.add_material(material2)
        materials = self.pool.list_materials()
        self.assertEqual(len(materials), 2)
    
    def test_list_materials_by_type(self):
        """Test listing materials filtered by type."""
        material1 = Material("Test1", MaterialType.DATASET)
        material2 = Material("Test2", MaterialType.MODEL)
        material3 = Material("Test3", MaterialType.DATASET)
        self.pool.add_material(material1)
        self.pool.add_material(material2)
        self.pool.add_material(material3)
        
        datasets = self.pool.list_materials(material_type=MaterialType.DATASET)
        self.assertEqual(len(datasets), 2)
        
        models = self.pool.list_materials(material_type=MaterialType.MODEL)
        self.assertEqual(len(models), 1)
    
    def test_get_materials_by_type(self):
        """Test getting materials by type."""
        material1 = Material("Test1", MaterialType.DATASET)
        material2 = Material("Test2", MaterialType.DATASET)
        self.pool.add_material(material1)
        self.pool.add_material(material2)
        
        datasets = self.pool.get_materials_by_type(MaterialType.DATASET)
        self.assertEqual(len(datasets), 2)
    
    def test_search_materials(self):
        """Test searching materials."""
        material1 = Material(
            "CNN Model",
            MaterialType.MODEL,
            description="Convolutional Neural Network"
        )
        material2 = Material(
            "LSTM Model",
            MaterialType.MODEL,
            description="Long Short-Term Memory"
        )
        material3 = Material(
            "Training Data",
            MaterialType.DATASET,
            description="CNN training dataset"
        )
        self.pool.add_material(material1)
        self.pool.add_material(material2)
        self.pool.add_material(material3)
        
        results = self.pool.search_materials("CNN")
        self.assertEqual(len(results), 2)
        
        results = self.pool.search_materials("LSTM")
        self.assertEqual(len(results), 1)
    
    def test_update_material(self):
        """Test updating a material."""
        material = Material("Test", MaterialType.DATASET)
        self.pool.add_material(material)
        result = self.pool.update_material("Test", description="Updated")
        self.assertTrue(result)
        
        updated = self.pool.get_material("Test")
        self.assertEqual(updated.description, "Updated")
    
    def test_update_nonexistent_material(self):
        """Test updating a nonexistent material."""
        result = self.pool.update_material("Nonexistent", description="Updated")
        self.assertFalse(result)
    
    def test_count(self):
        """Test counting materials."""
        self.assertEqual(self.pool.count(), 0)
        self.pool.add_material(Material("Test1", MaterialType.DATASET))
        self.assertEqual(self.pool.count(), 1)
        self.pool.add_material(Material("Test2", MaterialType.MODEL))
        self.assertEqual(self.pool.count(), 2)
    
    def test_clear(self):
        """Test clearing the pool."""
        self.pool.add_material(Material("Test1", MaterialType.DATASET))
        self.pool.add_material(Material("Test2", MaterialType.MODEL))
        self.pool.clear()
        self.assertEqual(self.pool.count(), 0)
    
    def test_export_to_json(self):
        """Test exporting to JSON."""
        material = Material(
            "Test",
            MaterialType.DATASET,
            description="Test dataset"
        )
        self.pool.add_material(material)
        json_str = self.pool.export_to_json()
        self.assertIn("Test", json_str)
        self.assertIn("dataset", json_str)
        self.assertIn("Test dataset", json_str)
    
    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.pool), 0)
        self.pool.add_material(Material("Test", MaterialType.DATASET))
        self.assertEqual(len(self.pool), 1)


if __name__ == "__main__":
    unittest.main()
