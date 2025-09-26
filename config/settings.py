"""
Configuration settings for the LlamaParse ingestion pipeline.

This module centralizes all configuration parameters and provides
type-safe configuration management.
"""

import os
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class LlamaParseConfig:
    """Configuration for LlamaParse API."""
    
    api_key: str
    base_url: Optional[str] = None
    result_type: str = "markdown"
    language: str = "en"
    num_workers: int = 1
    verbose: bool = True
    system_prompt: str = (
        "Focus on extracting structured data, tables, and key financial information. "
        "Preserve document hierarchy and formatting."
    )
    
    @classmethod
    def from_env(cls) -> 'LlamaParseConfig':
        """Create configuration from environment variables."""
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable is required")
        
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
            num_workers=int(os.getenv("LLAMA_PARSE_NUM_WORKERS", "1")),
            verbose=os.getenv("LLAMA_PARSE_VERBOSE", "true").lower() == "true"
        )


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    
    # Traditional chunking options
    chunk_size: int = 1500
    chunk_overlap: int = 300
    separators: List[str] = None
    keep_separator: bool = True
    
    # Semantic chunking options
    use_semantic_chunking: bool = False
    buffer_size: int = 1
    breakpoint_percentile_threshold: int = 95
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace default
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = [
                "\n## ",      # Split on markdown headers first
                "\n### ",     # Then subheaders
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ".",         # Sentences
                " ",         # Words
                ""           # Characters
            ]
        
        # No additional setup needed for HuggingFace embeddings


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings and vector store."""
    
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True
    show_progress_bar: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    
    docs_folder: str = "docs"
    output_folder: str = "output"
    vector_store_name: str = "vector_store_llamaparse"
    
    # Sub-configurations
    llamaparse: LlamaParseConfig = None
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    
    # Processing options
    save_results: bool = True
    test_retrieval: bool = True
    create_report: bool = True
    
    def __post_init__(self):
        if self.llamaparse is None:
            self.llamaparse = LlamaParseConfig.from_env()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
    
    @classmethod
    def create_default(
        cls,
        api_key: str,
        base_url: Optional[str] = None,
        docs_folder: str = "docs"
    ) -> 'PipelineConfig':
        """Create default configuration with minimal parameters."""
        llamaparse_config = LlamaParseConfig(
            api_key=api_key,
            base_url=base_url
        )
        
        return cls(
            docs_folder=docs_folder,
            llamaparse=llamaparse_config
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'docs_folder': self.docs_folder,
            'output_folder': self.output_folder,
            'vector_store_name': self.vector_store_name,
            'save_results': self.save_results,
            'test_retrieval': self.test_retrieval,
            'create_report': self.create_report,
            'llamaparse': {
                'api_key': self.llamaparse.api_key[:10] + "..." if self.llamaparse.api_key else None,  # Redact for security
                'base_url': self.llamaparse.base_url,
                'result_type': self.llamaparse.result_type,
                'language': self.llamaparse.language,
                'num_workers': self.llamaparse.num_workers,
                'verbose': self.llamaparse.verbose,
                'system_prompt': self.llamaparse.system_prompt
            },
            'chunking': {
                'chunk_size': self.chunking.chunk_size,
                'chunk_overlap': self.chunking.chunk_overlap,
                'keep_separator': self.chunking.keep_separator,
                'use_semantic_chunking': self.chunking.use_semantic_chunking,
                'buffer_size': self.chunking.buffer_size,
                'breakpoint_percentile_threshold': self.chunking.breakpoint_percentile_threshold,
                'embed_model_name': self.chunking.embed_model_name
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'device': self.embedding.device,
                'normalize_embeddings': self.embedding.normalize_embeddings,
                'show_progress_bar': self.embedding.show_progress_bar
            }
        }


# Default test queries for retrieval testing
DEFAULT_TEST_QUERIES = [
    "investment fund regulations and compliance requirements",
    "trading restrictions and market access rules", 
    "risk management framework and procedures",
    "financial reporting standards and disclosure requirements",
    "market data policies and data governance",
    "regulatory compliance and audit procedures"
]


# File type mappings
SUPPORTED_FILE_TYPES = {
    '.pdf': 'PDF Document',
    '.docx': 'Word Document', 
    '.pptx': 'PowerPoint Presentation',
    '.xlsx': 'Excel Spreadsheet',
    '.html': 'HTML Document'
}


# Environment variable names
ENV_VARS = {
    'API_KEY': 'LLAMA_CLOUD_API_KEY',
    'BASE_URL': 'LLAMA_CLOUD_BASE_URL',
    'NUM_WORKERS': 'LLAMA_PARSE_NUM_WORKERS',
    'VERBOSE': 'LLAMA_PARSE_VERBOSE'
}