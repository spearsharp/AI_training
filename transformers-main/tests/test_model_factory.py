"""Unit tests for model factory."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from models.model_factory import ModelFactory


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager("config/bert_classification.yaml").get_config()
        self.model_factory = ModelFactory(self.config)
    
    def test_create_classification_model(self):
        """Test creation of classification model."""
        model = self.model_factory.create_model()
        self.assertIsNotNone(model)
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model = self.model_factory.create_model()
        size_info = self.model_factory.get_model_size(model)
        
        self.assertIn('parameters', size_info)
        self.assertIn('trainable_parameters', size_info)
        self.assertIn('size_mb', size_info)
        self.assertGreater(size_info['parameters'], 0)


if __name__ == "__main__":
    unittest.main()