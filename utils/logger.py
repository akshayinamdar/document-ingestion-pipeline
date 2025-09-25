"""
Logger utilities with Windows compatibility for LlamaParse pipeline.

This module provides safe logging functionality that handles Unicode emojis
and provides consistent logging across the pipeline components.
"""

import sys
import logging
import re
from typing import Any


def safe_log_message(message: str) -> str:
    """
    Convert Unicode emojis to ASCII-safe alternatives for Windows compatibility.
    
    Args:
        message: The log message that may contain emojis
        
    Returns:
        ASCII-safe version of the message
    """
    emoji_map = {
        '🚀': '[>>]', '✅': '[OK]', '❌': '[ERR]', '🎉': '[SUCCESS]', 
        '⚠️': '[WARN]', '🔍': '[SEARCH]', '📄': '[DOC]', '📊': '[STATS]',
        '🔄': '[PROC]', '💾': '[SAVE]', '🔧': '[CONFIG]', '📁': '[FOLDER]',
        '🤖': '[AI]', '📈': '[GRAPH]', '📋': '[LIST]', '📝': '[NOTE]',
        '🧩': '[CHUNK]', '📏': '[SIZE]', '🔗': '[LINK]', '💻': '[CPU]',
        '🎯': '[TARGET]', '🗂️': '[FILES]', '⏱️': '[TIME]', '🌍': '[REGION]',
        '👥': '[WORKERS]', '📦': '[PACKAGE]', '🌐': '[WEB]', '💡': '[TIP]',
        '📂': '[OUTPUT]', '🏆': '[BEST]', '⭐': '[STAR]'
    }
    
    for emoji, replacement in emoji_map.items():
        message = message.replace(emoji, replacement)
    
    # Remove any remaining Unicode characters that might cause issues
    message = re.sub(r'[^\x00-\x7F]+', '[?]', message)
    return message


class SafeStreamHandler(logging.StreamHandler):
    """
    Custom stream handler that safely processes Unicode emojis for Windows compatibility.
    """
    
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = safe_log_message(msg)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class SafeLogger:
    """
    Logger wrapper that automatically handles emoji conversion for Windows compatibility.
    
    This wrapper ensures all log messages are processed through the emoji conversion
    function before being sent to the underlying logger.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the safe logger wrapper.
        
        Args:
            logger: The underlying Python logger instance
        """
        self._logger = logger
    
    def info(self, message: Any) -> None:
        """Log an info message with emoji conversion."""
        self._logger.info(safe_log_message(str(message)))
    
    def warning(self, message: Any) -> None:
        """Log a warning message with emoji conversion."""
        self._logger.warning(safe_log_message(str(message)))
    
    def error(self, message: Any) -> None:
        """Log an error message with emoji conversion."""
        self._logger.error(safe_log_message(str(message)))
    
    def debug(self, message: Any) -> None:
        """Log a debug message with emoji conversion."""
        self._logger.debug(safe_log_message(str(message)))


def setup_logging(
    log_file: str = "llamaparse_pipeline.log",
    log_level: int = logging.INFO,
    include_file_handler: bool = True
) -> SafeLogger:
    """
    Set up logging configuration for the LlamaParse pipeline.
    
    Args:
        log_file: Name of the log file to write to
        log_level: Logging level (default: INFO)
        include_file_handler: Whether to include file logging
        
    Returns:
        Configured SafeLogger instance
    """
    handlers = [SafeStreamHandler(sys.stdout)]
    
    if include_file_handler:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    return SafeLogger(logger)


def get_logger(name: str = __name__) -> SafeLogger:
    """
    Get a SafeLogger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        SafeLogger instance
    """
    return SafeLogger(logging.getLogger(name))