"""
Text preprocessing utilities.
"""

import re
import html
import string
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing for various NLP tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def preprocess(self, text: str) -> str:
        """Apply preprocessing steps to text."""
        if not isinstance(text, str):
            text = str(text)
            
        # Apply preprocessing steps based on configuration
        if self.config.get('lowercase', False):
            text = text.lower()
            
        if self.config.get('remove_html', False):
            text = self.remove_html_tags(text)
            
        if self.config.get('remove_urls', False):
            text = self.remove_urls(text)
            
        if self.config.get('remove_emails', False):
            text = self.remove_emails(text)
            
        if self.config.get('remove_phone_numbers', False):
            text = self.remove_phone_numbers(text)
            
        if self.config.get('normalize_whitespace', True):
            text = self.normalize_whitespace(text)
            
        if self.config.get('remove_punctuation', False):
            text = self.remove_punctuation(text)
            
        return text.strip()
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text."""
        # Decode HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.sub('', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove phone numbers from text."""
        phone_pattern = re.compile(r'(\+?1[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}')
        return phone_pattern.sub('', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    @staticmethod
    def clean_text(text: str) -> str:
        """General text cleaning function."""
        if not isinstance(text, str):
            text = str(text)
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


class TokenPreprocessor:
    """Token-level preprocessing utilities."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def add_special_tokens(self, text: str, add_bos: bool = False, add_eos: bool = False) -> str:
        """Add special tokens to text."""
        if add_bos and hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
            text = self.tokenizer.bos_token + text
            
        if add_eos and hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            text = text + self.tokenizer.eos_token
            
        return text
    
    def truncate_sequences(self, input_ids: list, max_length: int) -> list:
        """Truncate sequences to maximum length."""
        if len(input_ids) > max_length:
            return input_ids[:max_length]
        return input_ids