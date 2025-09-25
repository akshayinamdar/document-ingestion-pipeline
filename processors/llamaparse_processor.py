"""
LlamaParse document processor.

This module handles the core document processing using LlamaParse API,
including file discovery, parsing, and result management.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from llama_cloud_services import LlamaParse

from utils.logger import get_logger
from utils.helpers import get_pdf_files, format_file_size, ProgressTracker
from config.settings import LlamaParseConfig


class LlamaParseProcessor:
    """
    Handles document processing using LlamaParse API.
    
    This class manages the entire LlamaParse workflow from file discovery
    to parsed result collection.
    """
    
    def __init__(self, config: LlamaParseConfig):
        """
        Initialize the LlamaParse processor.
        
        Args:
            config: LlamaParse configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Set environment variables for LlamaParse
        os.environ["LLAMA_CLOUD_API_KEY"] = config.api_key
        if config.base_url:
            os.environ["LLAMA_CLOUD_BASE_URL"] = config.base_url
        
        # Processing state
        self.pdf_files: List[str] = []
        self.parsed_results: List[Dict[str, Any]] = []
        self.failed_files: List[tuple] = []
        
        # Statistics
        self.processing_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_pages': 0,
            'processing_time': 0
        }
        
        self.logger.info(f"[CONFIG] LlamaParse processor initialized")
        self.logger.info(f"[CONFIG] Result type: {config.result_type}")
        self.logger.info(f"[CONFIG] Workers: {config.num_workers}")
        self.logger.info(f"[CONFIG] Language: {config.language}")
        if config.base_url:
            self.logger.info(f"[REGION] Using base URL: {config.base_url}")
    
    def discover_documents(self, docs_folder: str) -> List[str]:
        """
        Discover PDF documents in the specified folder.
        
        Args:
            docs_folder: Path to folder containing documents
            
        Returns:
            List of discovered PDF file paths
        """
        self.logger.info(f"[SEARCH] Scanning '{docs_folder}' folder for PDF documents...")
        
        if not os.path.exists(docs_folder):
            self.logger.warning(f"[WARN] Warning: '{docs_folder}' folder not found!")
            self.logger.info("Please ensure your PDF documents are in the specified folder.")
            return []
        
        self.pdf_files = get_pdf_files(docs_folder)
        
        self.logger.info(f"[OK] Found {len(self.pdf_files)} PDF files:")
        for i, pdf_file in enumerate(self.pdf_files, 1):
            file_size = os.path.getsize(pdf_file)
            self.logger.info(f"  {i}. {os.path.basename(pdf_file)} ({format_file_size(file_size)})")
        
        self.processing_stats['total_files'] = len(self.pdf_files)
        return self.pdf_files
    
    def process_documents(self) -> bool:
        """
        Process all discovered PDF documents using LlamaParse.
        
        Returns:
            True if processing completed successfully (even with some failures)
        """
        if not self.pdf_files:
            self.logger.error("[ERR] No PDF files to process. Run discover_documents() first.")
            return False
        
        # Initialize LlamaParse parser
        self.logger.info("[CONFIG] Initializing LlamaParse parser...")
        
        try:
            parser = LlamaParse(
                api_key=self.config.api_key,
                result_type=self.config.result_type,
                num_workers=self.config.num_workers,
                verbose=self.config.verbose,
                language=self.config.language,
                system_prompt=self.config.system_prompt
            )
            self.logger.info("[OK] LlamaParse parser initialized")
        except Exception as e:
            self.logger.error(f"[ERR] Failed to initialize LlamaParse: {e}")
            return False
        
        self.logger.info("[>>] Starting LlamaParse document processing...")
        self.logger.info("=" * 60)
        
        # Reset processing state
        self.parsed_results = []
        self.failed_files = []
        
        start_time = datetime.now()
        progress = ProgressTracker(len(self.pdf_files), "Processing documents")
        
        # Process each PDF file
        for i, pdf_file in enumerate(self.pdf_files, 1):
            self.logger.info(f"\n[DOC] Processing file {i}/{len(self.pdf_files)}: {os.path.basename(pdf_file)}")
            self.logger.info("-" * 40)
            
            try:
                self.logger.info("  [PROC] Sending to LlamaParse API...")
                result = parser.parse(pdf_file)
                
                if result and result.pages:
                    self.logger.info(f"  [OK] Successfully parsed!")
                    self.logger.info(f"  [STATS] Pages processed: {len(result.pages)}")
                    
                    # Store result with metadata
                    self.parsed_results.append({
                        'file_path': pdf_file,
                        'file_name': os.path.basename(pdf_file),
                        'result': result,
                        'pages': len(result.pages),
                        'processed_at': datetime.now()
                    })
                    
                    self.processing_stats['successful'] += 1
                    self.processing_stats['total_pages'] += len(result.pages)
                    
                    # Show sample content
                    if result.pages and result.pages[0].md:
                        sample_content = result.pages[0].md[:200].strip()
                        self.logger.info(f"  [NOTE] Sample content: {sample_content}...")
                else:
                    self.logger.warning(f"  [WARN] Warning: No content extracted from {pdf_file}")
                    self.failed_files.append((pdf_file, "No content extracted"))
                    self.processing_stats['failed'] += 1
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"  [ERR] Error parsing {pdf_file}: {error_msg}")
                self.failed_files.append((pdf_file, error_msg))
                self.processing_stats['failed'] += 1
            
            progress.update()
        
        end_time = datetime.now()
        self.processing_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Display processing summary
        self._log_processing_summary()
        
        return True
    
    def _log_processing_summary(self) -> None:
        """Log processing summary statistics."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("[STATS] LLAMAPARSE PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"[OK] Successfully processed: {self.processing_stats['successful']} files")
        self.logger.info(f"[ERR] Failed to process: {self.processing_stats['failed']} files")
        self.logger.info(f"[DOC] Total pages extracted: {self.processing_stats['total_pages']}")
        self.logger.info(f"[TIME] Total processing time: {self.processing_stats['processing_time']:.2f} seconds")
        
        if self.failed_files:
            self.logger.warning("\n[ERR] Failed files:")
            for file_path, error in self.failed_files:
                self.logger.warning(f"  - {os.path.basename(file_path)}: {error}")
        
        if self.processing_stats['successful'] > 0:
            avg_time = self.processing_stats['processing_time'] / self.processing_stats['successful']
            self.logger.info(f"[GRAPH] Average time per file: {avg_time:.2f} seconds")
            
            avg_pages = self.processing_stats['total_pages'] / self.processing_stats['successful']
            self.logger.info(f"[GRAPH] Average pages per file: {avg_pages:.1f}")
        
        self.logger.info("[SUCCESS] LlamaParse processing complete!")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get the parsed results.
        
        Returns:
            List of parsed results with metadata
        """
        return self.parsed_results
    
    def get_failed_files(self) -> List[tuple]:
        """
        Get the list of failed files.
        
        Returns:
            List of (file_path, error_message) tuples
        """
        return self.failed_files
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return self.processing_stats.copy()
    
    def has_results(self) -> bool:
        """
        Check if there are any parsed results.
        
        Returns:
            True if there are parsed results
        """
        return len(self.parsed_results) > 0