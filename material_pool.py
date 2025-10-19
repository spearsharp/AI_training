"""
Material Pool for AI Training Project

This module provides a material pool system for managing AI training resources.
It allows organizing, storing, and retrieving various types of materials such as
datasets, models, documentation, and other resources.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class MaterialType(Enum):
    """Types of materials that can be stored in the pool."""
    DATASET = "dataset"
    MODEL = "model"
    DOCUMENTATION = "documentation"
    SCRIPT = "script"
    CONFIG = "config"
    OTHER = "other"


class Material:
    """Represents a single material in the pool."""
    
    def __init__(
        self,
        name: str,
        material_type: MaterialType,
        description: str = "",
        path: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Material.
        
        Args:
            name: Name of the material
            material_type: Type of the material
            description: Description of the material
            path: Path to the material resource
            metadata: Additional metadata about the material
        """
        self.name = name
        self.material_type = material_type
        self.description = description
        self.path = path
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update(self, **kwargs):
        """Update material attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert material to dictionary."""
        return {
            "name": self.name,
            "material_type": self.material_type.value,
            "description": self.description,
            "path": self.path,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def __repr__(self) -> str:
        return f"Material(name='{self.name}', type={self.material_type.value})"


class MaterialPool:
    """
    A pool for managing AI training materials.
    
    Provides functionality to add, retrieve, list, and remove materials
    from the pool.
    """
    
    def __init__(self):
        """Initialize an empty material pool."""
        self._materials: Dict[str, Material] = {}
    
    def add_material(self, material: Material) -> bool:
        """
        Add a material to the pool.
        
        Args:
            material: The material to add
            
        Returns:
            True if material was added, False if material with same name exists
        """
        if material.name in self._materials:
            return False
        self._materials[material.name] = material
        return True
    
    def get_material(self, name: str) -> Optional[Material]:
        """
        Retrieve a material by name.
        
        Args:
            name: Name of the material
            
        Returns:
            The material if found, None otherwise
        """
        return self._materials.get(name)
    
    def remove_material(self, name: str) -> bool:
        """
        Remove a material from the pool.
        
        Args:
            name: Name of the material to remove
            
        Returns:
            True if material was removed, False if not found
        """
        if name in self._materials:
            del self._materials[name]
            return True
        return False
    
    def list_materials(
        self,
        material_type: Optional[MaterialType] = None
    ) -> List[Material]:
        """
        List all materials in the pool.
        
        Args:
            material_type: Optional filter by material type
            
        Returns:
            List of materials
        """
        materials = list(self._materials.values())
        if material_type:
            materials = [m for m in materials if m.material_type == material_type]
        return materials
    
    def update_material(self, name: str, **kwargs) -> bool:
        """
        Update a material's attributes.
        
        Args:
            name: Name of the material
            **kwargs: Attributes to update
            
        Returns:
            True if material was updated, False if not found
        """
        material = self.get_material(name)
        if material:
            material.update(**kwargs)
            return True
        return False
    
    def get_materials_by_type(self, material_type: MaterialType) -> List[Material]:
        """
        Get all materials of a specific type.
        
        Args:
            material_type: The type to filter by
            
        Returns:
            List of materials of the specified type
        """
        return self.list_materials(material_type=material_type)
    
    def search_materials(self, query: str) -> List[Material]:
        """
        Search materials by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching materials
        """
        query_lower = query.lower()
        return [
            m for m in self._materials.values()
            if query_lower in m.name.lower() or query_lower in m.description.lower()
        ]
    
    def count(self) -> int:
        """Get the total number of materials in the pool."""
        return len(self._materials)
    
    def clear(self):
        """Remove all materials from the pool."""
        self._materials.clear()
    
    def export_to_json(self) -> str:
        """
        Export the material pool to JSON.
        
        Returns:
            JSON string representation of the pool
        """
        materials_dict = {
            name: material.to_dict()
            for name, material in self._materials.items()
        }
        return json.dumps(materials_dict, indent=2)
    
    def __len__(self) -> int:
        return len(self._materials)
    
    def __repr__(self) -> str:
        return f"MaterialPool(materials={len(self._materials)})"
