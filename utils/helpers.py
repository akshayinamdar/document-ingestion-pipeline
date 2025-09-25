"""
Helper utilities for the LlamaParse ingestion pipeline.

This module contains common utility functions used across the pipeline.
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted size string (e.g., "2.5 KB", "1.2 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_directory(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
        
    Returns:
        Absolute path to directory
    """
    abs_path = os.path.abspath(directory)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data as JSON file with proper error handling.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON to {file_path}: {e}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with proper error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {file_path}: {e}")


def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data as pickle file with proper error handling.
    
    Args:
        data: Data to pickle
        file_path: Path to save the pickle file
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise RuntimeError(f"Failed to save pickle to {file_path}: {e}")


def load_pickle(file_path: str) -> Any:
    """
    Load pickle file with proper error handling.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded pickle data
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle from {file_path}: {e}")


def get_timestamp() -> str:
    """
    Get current timestamp in standard format.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_pdf_files(directory: str) -> List[str]:
    """
    Get list of PDF files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of PDF file paths
    """
    if not os.path.exists(directory):
        return []
    
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    
    return sorted(pdf_files)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics
    """
    if not values:
        return {
            'count': 0,
            'sum': 0,
            'mean': 0,
            'min': 0,
            'max': 0
        }
    
    return {
        'count': len(values),
        'sum': sum(values),
        'mean': sum(values) / len(values),
        'min': min(values),
        'max': max(values)
    }


def create_processing_report(
    session_id: str,
    processing_stats: Dict[str, Any],
    parsed_results: List[Dict[str, Any]],
    failed_files: List[tuple],
    all_documents: List[Any],
    chunked_documents: List[Any],
    config: Dict[str, Any]
) -> str:
    """
    Create a markdown processing report.
    
    Args:
        session_id: Unique session identifier
        processing_stats: Processing statistics
        parsed_results: List of successfully parsed results
        failed_files: List of failed files with errors
        all_documents: List of all documents
        chunked_documents: List of chunked documents
        config: Pipeline configuration
        
    Returns:
        Markdown report content
    """
    report_lines = [
        f"# LlamaParse Processing Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"## Summary",
        f"- **Session ID**: {session_id}",
        f"- **Files Processed**: {processing_stats.get('successful', 0)}/{processing_stats.get('total_files', 0)}",
        f"- **Total Pages**: {processing_stats.get('total_pages', 0)}",
        f"- **Documents Created**: {len(all_documents)}",
        f"- **Chunks Generated**: {len(chunked_documents)}",
        f"- **Processing Time**: {processing_stats.get('processing_time', 0):.2f} seconds",
        "",
    ]
    
    if parsed_results:
        report_lines.extend([
            f"## Successfully Processed Files",
        ])
        for result in parsed_results:
            report_lines.append(f"- {result['file_name']} ({result['pages']} pages)")
    
    if failed_files:
        report_lines.extend([
            "",
            f"## Failed Files",
        ])
        for file_path, error in failed_files:
            filename = os.path.basename(file_path)
            report_lines.append(f"- {filename}: {error}")
    
    report_lines.extend([
        "",
        f"## Configuration",
        f"- **Parser**: LlamaParse (markdown output)",
        f"- **Chunking**: {len(chunked_documents)} chunks",
        f"- **Embeddings**: {config.get('embedding_model', 'N/A')}",
        f"- **Vector Store**: FAISS",
        f"- **Chunk Size**: {config.get('chunk_size', 'N/A')} characters",
        f"- **Chunk Overlap**: {config.get('chunk_overlap', 'N/A')} characters",
        "",
        f"## Pipeline Configuration",
    ])
    
    for key, value in config.items():
        report_lines.append(f"- **{key}**: {value}")
    
    return "\n".join(report_lines)


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    """
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress.
        
        Args:
            increment: Number of items processed
        """
        self.current_item += increment
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary with progress info
        """
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        progress_pct = (self.current_item / self.total_items * 100) if self.total_items > 0 else 0
        
        return {
            'current': self.current_item,
            'total': self.total_items,
            'percentage': progress_pct,
            'elapsed_time': elapsed_time,
            'description': self.description
        }