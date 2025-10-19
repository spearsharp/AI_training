"""Unit tests for data loader."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from data.data_loader import CustomDataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for CustomDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigManager("config/bert_classification.yaml").get_config()
        self.data_loader = CustomDataLoader(self.config)
    
    def test_tokenizer_initialization(self):
        """Test tokenizer is properly initialized."""
        self.assertIsNotNone(self.data_loader.tokenizer)
    
    def test_load_datasets(self):
        """Test dataset loading."""
        # This would require actual data or mocking
        pass
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        # Test preprocessing functionality
        pass


class TestPreprocessor(unittest.TestCase):
    """Test cases for text preprocessing."""
    
    def test_remove_html(self):
        """Test HTML tag removal."""
        from data.preprocessor import TextPreprocessor
        
        config = {'remove_html': True}
        preprocessor = TextPreprocessor(config)
        
        text = "<p>Hello <b>world</b>!</p>"
        cleaned = preprocessor.remove_html_tags(text)
        
        self.assertEqual(cleaned, "Hello world!")
    
    def test_remove_urls(self):
        """Test URL removal."""
        from data.preprocessor import TextPreprocessor
        
        config = {'remove_urls': True}
        preprocessor = TextPreprocessor(config)
        
        text = "Check out https://example.com for more info"
        cleaned = preprocessor.remove_urls(text)
        
        self.assertNotIn("https://example.com", cleaned)


if __name__ == "__main__":
    unittest.main()