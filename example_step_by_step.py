#!/usr/bin/env python3
"""
Example: Using the modular LlamaParse pipeline step-by-step

This example demonstrates how to use the individual components
of the pipeline for more control over the process.
"""

from config.settings import PipelineConfig
from processors.llamaparse_processor import LlamaParseProcessor
from processors.document_converter import DocumentConverter
from stores.vector_store import VectorStoreManager
from utils.logger import setup_logging


def main():
    """Example of step-by-step pipeline usage."""
    
    # Setup logging
    logger = setup_logging()
    logger.info("🔧 Running step-by-step pipeline example")
    
    # Create configuration
    config = PipelineConfig.create_default(
        api_key="llx-MHXNHhQO6ahnPlDx7i5O3QYgdYdaVmhtWIFyrJPI1zszoSu8",
        base_url="https://api.cloud.eu.llamaindex.ai",
        docs_folder="docs"
    )
    
    # Initialize components individually
    processor = LlamaParseProcessor(config.llamaparse)
    converter = DocumentConverter(config.chunking)
    vector_manager = VectorStoreManager(config.embedding)
    
    try:
        # Step 1: Discover and process documents
        logger.info("🔍 Step 1: Discovering documents...")
        pdf_files = processor.discover_documents(config.docs_folder)
        
        if not pdf_files:
            logger.error("No PDF files found!")
            return False
        
        logger.info("📄 Step 2: Processing with LlamaParse...")
        if not processor.process_documents():
            logger.error("Processing failed!")
            return False
        
        # Step 2: Convert to documents
        logger.info("🔄 Step 3: Converting to LangChain documents...")
        parsed_results = processor.get_results()
        documents = converter.convert_to_documents(parsed_results)
        
        if not documents:
            logger.error("Document conversion failed!")
            return False
        
        # Step 3: Chunk documents
        logger.info("🧩 Step 4: Chunking documents...")
        chunks = converter.chunk_documents(documents)
        
        if not chunks:
            logger.error("Document chunking failed!")
            return False
        
        # Step 4: Create vector store
        logger.info("🤖 Step 5: Creating vector store...")
        if not vector_manager.create_vector_store(chunks):
            logger.error("Vector store creation failed!")
            return False
        
        # Step 5: Test search
        logger.info("🔍 Step 6: Testing search...")
        test_query = "financial regulations and compliance"
        results = vector_manager.similarity_search(test_query, k=3)
        
        logger.info(f"Found {len(results)} results for: '{test_query}'")
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('file_name', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            logger.info(f"  {i}. {source} (Page {page})")
        
        # Step 6: Save vector store
        logger.info("💾 Step 7: Saving vector store...")
        if vector_manager.save_vector_store("output/example_vector_store"):
            logger.info("✅ Vector store saved successfully!")
        
        logger.info("🎉 Step-by-step example completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in step-by-step example: {e}")
        return False


if __name__ == "__main__":
    main()