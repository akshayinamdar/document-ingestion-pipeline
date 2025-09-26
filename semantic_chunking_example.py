"""
Example: Semantic Chunking with LlamaIndex

This example demonstrates how to use semantic chunking instead of 
traditional fixed-size chunking for better RAG performance.
"""

import os
from dotenv import load_dotenv

from config.settings import PipelineConfig, ChunkingConfig
from processors.llamaparse_processor import LlamaParseProcessor
from processors.document_converter import DocumentConverter
from stores.vector_store import VectorStoreManager
from utils.logger import get_logger

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate semantic chunking."""
    
    logger = get_logger(__name__)
    
    # Verify required API key
    llama_api_key = os.getenv('LLAMA_CLOUD_API_KEY')
    
    if not llama_api_key:
        logger.error("LLAMA_CLOUD_API_KEY not found. Please set it in your .env file.")
        return
    
    # Configure semantic chunking
    semantic_chunking_config = ChunkingConfig(
        # Enable semantic chunking
        use_semantic_chunking=True,
        
        # Semantic parameters - adjust based on your needs
        buffer_size=1,  # Consider 1 sentence at a time
        breakpoint_percentile_threshold=95,  # 95th percentile for breakpoints
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",  # HuggingFace embedding model
        
        # Fallback traditional chunking parameters
        chunk_size=1500,
        chunk_overlap=300
    )
    
    # Create pipeline configuration
    config = PipelineConfig(
        docs_folder="docs",
        output_folder="output",
        vector_store_name="semantic_chunks_vector_store",
        chunking=semantic_chunking_config
    )
    
    logger.info("Starting semantic chunking pipeline...")
    logger.info(f"Semantic chunking enabled: {config.chunking.use_semantic_chunking}")
    
    # Initialize processors
    llamaparse_processor = LlamaParseProcessor(config.llamaparse)
    document_converter = DocumentConverter(config.chunking)
    vector_store = VectorStoreManager(config.embedding)
    
    # Check if documents exist
    docs_folder = config.docs_folder
    if not os.path.exists(docs_folder) or not os.listdir(docs_folder):
        logger.warning(f"No documents found in {docs_folder}. Please add some PDF, DOCX, or other supported files.")
        return
    
    try:
        # Step 1: Discover and parse documents with LlamaParse
        logger.info("Step 1: Discovering documents...")
        pdf_files = llamaparse_processor.discover_documents(docs_folder)
        
        if not pdf_files:
            logger.error("No PDF files found to process.")
            return
        
        logger.info("Step 1b: Parsing documents with LlamaParse...")
        if not llamaparse_processor.process_documents():
            logger.error("Document processing failed.")
            return
        
        parsed_results = llamaparse_processor.get_results()
        if not parsed_results:
            logger.error("No documents were successfully parsed.")
            return
        
        # Step 2: Convert to LangChain documents
        logger.info("Step 2: Converting to LangChain documents...")
        documents = document_converter.convert_to_documents(parsed_results)
        
        # Step 3: Apply semantic chunking
        logger.info("Step 3: Applying semantic chunking...")
        chunks = document_converter.chunk_documents(documents)
        
        # Step 4: Create vector store
        logger.info("Step 4: Creating vector embeddings...")
        if vector_store.create_vector_store(chunks):
            # Save the vector store
            save_path = os.path.join(config.output_folder, config.vector_store_name)
            os.makedirs(config.output_folder, exist_ok=True)
            vector_store.save_vector_store(save_path)
            vector_store_path = save_path
        else:
            logger.error("Failed to create vector store")
            return
        
        logger.info("=" * 60)
        logger.info("SEMANTIC CHUNKING PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Chunks created: {len(chunks)}")
        logger.info(f"Vector store: {vector_store_path}")
        
        # Show chunking method used
        if chunks:
            chunking_method = chunks[0].metadata.get('chunking_method', 'unknown')
            logger.info(f"Chunking method: {chunking_method}")
            
            if chunking_method == 'semantic':
                logger.info(f"Semantic threshold: {chunks[0].metadata.get('semantic_threshold')}%")
                logger.info(f"Buffer size: {chunks[0].metadata.get('buffer_size')}")
                logger.info(f"Embedding model: {chunks[0].metadata.get('embedding_model')}")
                logger.info(f"Embedding provider: {chunks[0].metadata.get('embedding_provider')}")
        
        # Test retrieval (optional)
        if config.test_retrieval:
            logger.info("\nTesting retrieval with semantic chunks...")
            test_queries = [
                "What is the main topic discussed in the documents?",
                "Are there any key recommendations or conclusions?",
                "What are the important details mentioned?"
            ]
            
            for query in test_queries:
                logger.info(f"\nQuery: {query}")
                results = vector_store.similarity_search(query, k=3)
                
                for i, doc in enumerate(results[:2]):  # Show top 2 results
                    chunk_method = doc.metadata.get('chunking_method', 'unknown')
                    logger.info(f"  Result {i+1} ({chunk_method}): {doc.page_content[:100]}...")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()