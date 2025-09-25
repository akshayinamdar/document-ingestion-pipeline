"""
LlamaParse Document Ingestion Pipeline

A modular, production-ready document ingestion pipeline using LlamaParse.
Transforms PDF documents into searchable vector stores for RAG applications.

Main Components:
- LlamaParseProcessor: Document parsing with LlamaParse API
- DocumentConverter: Document conversion and chunking
- VectorStoreManager: Embedding generation and vector store management
- IngestionPipeline: Complete pipeline orchestration

Usage:
    from pipeline import IngestionPipeline
    from config import PipelineConfig
    
    config = PipelineConfig.create_default(api_key="your-key")
    pipeline = IngestionPipeline(config)
    success = pipeline.run_complete_pipeline()
"""

from .pipeline import IngestionPipeline
from .config import PipelineConfig, LlamaParseConfig, ChunkingConfig, EmbeddingConfig
from .processors import LlamaParseProcessor, DocumentConverter
from .stores import VectorStoreManager
from .utils import get_logger, setup_logging

__version__ = "1.0.0"
__author__ = "Document Ingestion Team"

__all__ = [
    'IngestionPipeline',
    'PipelineConfig', 'LlamaParseConfig', 'ChunkingConfig', 'EmbeddingConfig',
    'LlamaParseProcessor', 'DocumentConverter', 'VectorStoreManager',
    'get_logger', 'setup_logging'
]