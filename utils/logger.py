"""
Logging Utilities
"""
import logging
import os
import sys
from typing import Optional
from logging.handlers import RotatingFileHandler

# Global logger cache
_LOGGERS = {}

def setup_logger() -> logging.Logger:
    """Set up and configure the root logger."""
    # Save logs in user home directory (cross-platform)
    log_dir = os.path.join(os.path.expanduser("~"), "multilllm_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Determine log level from environment or default to INFO
    log_level_name = os.environ.get('LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Create file handler
    log_file_path = os.path.join(log_dir, 'multilllm.log')
    file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    return root_logger
log_dir = os.environ.get('LOG_DIR', os.path.join(os.path.expanduser("~"), "multilllm_logs"))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = ''
    
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    logger = logging.getLogger(name)
    
    # Only add handlers if not present
    if not logger.handlers:
        setup_logger()  # Optional safeguard if not called before

    _LOGGERS[name] = logger
    return logger
