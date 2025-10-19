"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logger(
    name: str = "transformers-main",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    Setup logger with optional file output and rich formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        use_rich: Whether to use rich formatting
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
    
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars."""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)