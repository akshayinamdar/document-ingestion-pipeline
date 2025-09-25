#!/usr/bin/env python3
"""
LlamaParse Document Ingestion Pipeline - Main Entry Point

A modular, production-ready document ingestion pipeline using LlamaParse.
Transforms PDF documents into searchable vector stores for RAG applications.

Usage:
    python main.py

Environment Variables:
    LLAMA_CLOUD_API_KEY: Your LlamaParse API key (required)
    LLAMA_CLOUD_BASE_URL: Optional, for EU region use https://api.cloud.eu.llamaindex.ai

Features:
    - ✅ Modular architecture with clear separation of concerns
    - ✅ Type-safe configuration management
    - ✅ Comprehensive error handling and logging
    - ✅ Windows emoji compatibility
    - ✅ Production-ready with statistics and monitoring
    - ✅ Easily extensible and testable
"""

import sys
from typing import Optional

# Import our modular components
from pipeline.ingestion_pipeline import IngestionPipeline
from config.settings import PipelineConfig
from utils.logger import setup_logging

# Import dependencies with graceful error handling
try:
    from llama_cloud_services import LlamaParse
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install required packages:")
    print("pip install llama-cloud-services langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers")
    sys.exit(1)


def create_default_config() -> Optional[PipelineConfig]:
    """
    Create default pipeline configuration.
    
    Returns:
        PipelineConfig object or None if API key is missing
    """
    # Configuration - you can modify these values
    API_KEY = "llx-MHXNHhQO6ahnPlDx7i5O3QYgdYdaVmhtWIFyrJPI1zszoSu8"
    BASE_URL = "https://api.cloud.eu.llamaindex.ai"  # EU region
    DOCS_FOLDER = "docs"
    
    try:
        return PipelineConfig.create_default(
            api_key=API_KEY,
            base_url=BASE_URL,
            docs_folder=DOCS_FOLDER
        )
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set LLAMA_CLOUD_API_KEY environment variable or update the API_KEY in main.py")
        return None


def main():
    """
    Main function to run the LlamaParse ingestion pipeline.
    """
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 Starting LlamaParse Document Ingestion Pipeline")
    logger.info("=" * 70)
    
    try:
        # Create configuration
        config = create_default_config()
        if not config:
            sys.exit(1)
        
        logger.info("✅ Configuration loaded successfully")
        
        # Initialize pipeline
        pipeline = IngestionPipeline(config)
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
            
            # Show session information
            logger.info(f"\n📋 Session Information:")
            logger.info(f"  - Session ID: {pipeline.get_session_id()}")
            logger.info(f"  - Documents processed: {len(pipeline.get_documents())}")
            logger.info(f"  - Chunks created: {len(pipeline.get_chunks())}")
            logger.info(f"  - Vector store ready: {'Yes' if pipeline.has_vector_store() else 'No'}")
            
            sys.exit(0)
        else:
            logger.error("❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error("Please check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()