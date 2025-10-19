"""
Configuration manager for handling YAML config files.
"""

import yaml
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf for dot notation access
        config = OmegaConf.create(config_dict)
        
        # Validate required fields
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: DictConfig) -> None:
        """Validate that required configuration fields are present."""
        required_sections = ['model', 'data', 'training']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get_config(self) -> DictConfig:
        """Get the loaded configuration."""
        return self.config
    
    def get_section(self, section_name: str) -> DictConfig:
        """Get a specific configuration section."""
        if section_name not in self.config:
            raise ValueError(f"Config section not found: {section_name}")
        return self.config[section_name]
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config = OmegaConf.merge(self.config, updates)
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to file."""
        with open(output_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(self.config), f, default_flow_style=False)