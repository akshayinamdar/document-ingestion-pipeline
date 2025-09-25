"""
Main ingestion pipeline orchestrator.

This module coordinates all pipeline components to provide a complete
document ingestion workflow from PDFs to searchable vector store.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from processors.llamaparse_processor import LlamaParseProcessor
from processors.document_converter import DocumentConverter
from stores.vector_store import VectorStoreManager
from config.settings import PipelineConfig
from utils.logger import get_logger
from utils.helpers import (
    ensure_directory, save_json, save_pickle, get_timestamp, 
    create_processing_report
)


class IngestionPipeline:
    """
    Complete document ingestion pipeline orchestrator.
    
    This class coordinates all pipeline components to transform PDF documents
    into a searchable vector store suitable for RAG applications.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.processor = LlamaParseProcessor(config.llamaparse)
        self.converter = DocumentConverter(config.chunking)
        self.vector_manager = VectorStoreManager(config.embedding)
        
        # Session information
        self.session_id = f"llamaparse_{get_timestamp()}"
        
        self.logger.info("[>>] LlamaParse Document Ingestion Pipeline Initialized")
        self.logger.info(f"[CONFIG] Session ID: {self.session_id}")
        self.logger.info(f"[FOLDER] Documents folder: {config.docs_folder}")
        self.logger.info(f"[OUTPUT] Output folder: {config.output_folder}")
        
        # Log configuration summary
        self._log_configuration_summary()
    
    def _log_configuration_summary(self) -> None:
        """Log a summary of the current configuration."""
        self.logger.info(f"[CONFIG] Chunk size: {self.config.chunking.chunk_size}, Overlap: {self.config.chunking.chunk_overlap}")
        self.logger.info(f"[WORKERS] Workers: {self.config.llamaparse.num_workers}")
        self.logger.info(f"[AI] Embedding model: {self.config.embedding.model_name}")
        
        if self.config.llamaparse.base_url:
            self.logger.info(f"[REGION] Using region: {self.config.llamaparse.base_url}")
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete ingestion pipeline from start to finish.
        
        Returns:
            True if pipeline completed successfully
        """
        self.logger.info("[>>] Starting Complete LlamaParse Ingestion Pipeline")
        self.logger.info("=" * 70)
        
        try:
            # Step 1: Discover documents
            self.logger.info("\n[STEP 1] Document Discovery")
            if not self._discover_documents():
                return False
            
            # Step 2: Process documents with LlamaParse
            self.logger.info("\n[STEP 2] LlamaParse Processing")
            if not self._process_documents():
                return False
            
            # Step 3: Convert to LangChain documents
            self.logger.info("\n[STEP 3] Document Conversion")
            if not self._convert_documents():
                return False
            
            # Step 4: Chunk documents
            self.logger.info("\n[STEP 4] Document Chunking")
            if not self._chunk_documents():
                return False
            
            # Step 5: Create vector store
            self.logger.info("\n[STEP 5] Vector Store Creation")
            if not self._create_vector_store():
                return False
            
            # Step 6: Test retrieval (optional)
            if self.config.test_retrieval:
                self.logger.info("\n[STEP 6] Retrieval Testing")
                self._test_retrieval()
            
            # Step 7: Save results (optional)
            if self.config.save_results:
                self.logger.info("\n[STEP 7] Save Results")
                saved_files = self._save_results()
                
                if saved_files:
                    self.logger.info(f"\n[OUTPUT] Output Files:")
                    for result_type, file_path in saved_files.items():
                        self.logger.info(f"  - {result_type}: {file_path}")
            
            # Final summary
            self._log_final_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERR] Pipeline failed with error: {e}")
            return False
    
    def _discover_documents(self) -> bool:
        """Discover documents step."""
        documents = self.processor.discover_documents(self.config.docs_folder)
        if not documents:
            self.logger.error("[ERR] No documents found to process")
            return False
        return True
    
    def _process_documents(self) -> bool:
        """Process documents step."""
        if not self.processor.process_documents():
            self.logger.error("[ERR] Document processing failed")
            return False
        
        if not self.processor.has_results():
            self.logger.error("[ERR] No documents were successfully processed")
            return False
        
        return True
    
    def _convert_documents(self) -> bool:
        """Convert documents step."""
        parsed_results = self.processor.get_results()
        documents = self.converter.convert_to_documents(parsed_results)
        
        if not documents:
            self.logger.error("[ERR] Document conversion failed")
            return False
        
        return True
    
    def _chunk_documents(self) -> bool:
        """Chunk documents step."""
        chunks = self.converter.chunk_documents()
        
        if not chunks:
            self.logger.error("[ERR] Document chunking failed")
            return False
        
        return True
    
    def _create_vector_store(self) -> bool:
        """Create vector store step."""
        chunks = self.converter.get_chunks()
        
        if not self.vector_manager.create_vector_store(chunks):
            self.logger.error("[ERR] Vector store creation failed")
            return False
        
        return True
    
    def _test_retrieval(self) -> bool:
        """Test retrieval step."""
        return self.vector_manager.test_retrieval()
    
    def _save_results(self) -> Dict[str, str]:
        """Save results step."""
        output_dir = ensure_directory(self.config.output_folder)
        timestamp = get_timestamp()
        saved_files = {}
        
        try:
            # Save vector store
            vector_store_path = os.path.join(output_dir, self.config.vector_store_name)
            if self.vector_manager.save_vector_store(vector_store_path):
                saved_files['vector_store'] = vector_store_path
            
            # Save processed documents
            chunks = self.converter.get_chunks()
            if chunks:
                documents_file = os.path.join(output_dir, f"processed_documents_{timestamp}.pkl")
                save_pickle(chunks, documents_file)
                saved_files['documents'] = documents_file
            
            # Save configuration
            config_file = os.path.join(output_dir, f"pipeline_config_{timestamp}.json")
            config_data = {
                'session_id': self.session_id,
                'timestamp': timestamp,
                **self.config.to_dict(),
                'processing_stats': self.processor.get_statistics(),
                'files_processed': [r['file_name'] for r in self.processor.get_results()],
                'failed_files': [(os.path.basename(f), e) for f, e in self.processor.get_failed_files()]
            }
            save_json(config_data, config_file)
            saved_files['config'] = config_file
            
            # Save raw results (skip due to thread lock serialization issues)
            # Note: Raw results contain LlamaParse objects with thread locks that can't be pickled
            # For debugging, use the processing report and config files instead
            
            # Create processing report
            if self.config.create_report:
                report_content = create_processing_report(
                    session_id=self.session_id,
                    processing_stats=self.processor.get_statistics(),
                    parsed_results=self.processor.get_results(),
                    failed_files=self.processor.get_failed_files(),
                    all_documents=self.converter.get_documents(),
                    chunked_documents=self.converter.get_chunks(),
                    config=self.config.to_dict()
                )
                
                report_file = os.path.join(output_dir, f"processing_report_{timestamp}.md")
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                saved_files['report'] = report_file
            
            self.logger.info("[SUCCESS] All pipeline results saved successfully!")
            
        except Exception as e:
            self.logger.error(f"[ERR] Error saving results: {e}")
        
        return saved_files
    
    def _log_final_summary(self) -> None:
        """Log final pipeline summary."""
        stats = self.processor.get_statistics()
        documents = self.converter.get_documents()
        chunks = self.converter.get_chunks()
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("[SUCCESS] LLAMAPARSE INGESTION PIPELINE COMPLETE!")
        self.logger.info("=" * 70)
        self.logger.info(f"[OK] Successfully processed {stats['successful']} PDF files")
        self.logger.info(f"[OK] Generated {stats['total_pages']} pages of structured content")
        self.logger.info(f"[OK] Created {len(documents)} LangChain documents")
        self.logger.info(f"[OK] Split into {len(chunks)} optimized chunks")
        self.logger.info(f"[OK] Built vector store with 384-dimensional embeddings")
        self.logger.info(f"[OK] Total processing time: {stats['processing_time']:.2f} seconds")
        
        self.logger.info(f"\n[>>] Ready for RAG Implementation!")
        self.logger.info(f"Your vector store is ready for use in RAG pipelines.")
    
    # Public API methods for individual steps
    
    def discover_documents(self) -> List[str]:
        """
        Discover documents in the configured folder.
        
        Returns:
            List of discovered PDF file paths
        """
        return self.processor.discover_documents(self.config.docs_folder)
    
    def process_documents(self) -> bool:
        """
        Process discovered documents with LlamaParse.
        
        Returns:
            True if processing was successful
        """
        return self.processor.process_documents()
    
    def convert_to_documents(self) -> List[Any]:
        """
        Convert LlamaParse results to LangChain documents.
        
        Returns:
            List of LangChain Document objects
        """
        parsed_results = self.processor.get_results()
        return self.converter.convert_to_documents(parsed_results)
    
    def chunk_documents(self) -> List[Any]:
        """
        Chunk documents for RAG optimization.
        
        Returns:
            List of chunked Document objects
        """
        return self.converter.chunk_documents()
    
    def create_vector_store(self) -> bool:
        """
        Create vector store from chunked documents.
        
        Returns:
            True if vector store creation was successful
        """
        chunks = self.converter.get_chunks()
        return self.vector_manager.create_vector_store(chunks)
    
    def test_retrieval(self, queries: Optional[List[str]] = None) -> bool:
        """
        Test document retrieval with sample queries.
        
        Args:
            queries: Optional custom queries to test
            
        Returns:
            True if testing was successful
        """
        return self.vector_manager.test_retrieval(queries)
    
    def search(self, query: str, k: int = 4) -> List[Any]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        return self.vector_manager.similarity_search(query, k)
    
    # Getters for component results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processor.get_statistics()
    
    def get_parsed_results(self) -> List[Dict[str, Any]]:
        """Get raw parsed results."""
        return self.processor.get_results()
    
    def get_failed_files(self) -> List[tuple]:
        """Get list of failed files."""
        return self.processor.get_failed_files()
    
    def get_documents(self) -> List[Any]:
        """Get converted documents."""
        return self.converter.get_documents()
    
    def get_chunks(self) -> List[Any]:
        """Get chunked documents."""
        return self.converter.get_chunks()
    
    def has_vector_store(self) -> bool:
        """Check if vector store is available."""
        return self.vector_manager.has_vector_store()
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id