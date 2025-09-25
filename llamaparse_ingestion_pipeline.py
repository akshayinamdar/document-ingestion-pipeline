#!/usr/bin/env python3
"""
LlamaParse Document Ingestion Pipeline

This script implements a complete document ingestion pipeline using LlamaParse,
a GenAI-native document parser that excels at parsing complex documents with
tables, visual elements, and varied layouts.

Key Features:
- ✅ Broad file type support: .pdf, .pptx, .docx, .xlsx, .html
- ✅ Table recognition: Accurate parsing of embedded tables  
- ✅ Multimodal parsing: Extraction of visual elements
- ✅ Custom parsing: Configurable output formatting

Pipeline Steps:
1. Setup: Configure LlamaParse API and environment
2. Discovery: Scan for PDF documents in the docs folder
3. Parsing: Process documents using LlamaParse API
4. Conversion: Transform results to LangChain Document format
5. Chunking: Split documents into manageable chunks
6. Vectorization: Create embeddings and vector store for RAG

Usage:
    python llamaparse_ingestion_pipeline.py

Environment Variables:
    LLAMA_CLOUD_API_KEY: Your LlamaParse API key
    LLAMA_CLOUD_BASE_URL: Optional, for EU region use https://api.cloud.eu.llamaindex.ai
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re

def safe_log_message(message: str) -> str:
    """Convert Unicode emojis to ASCII-safe alternatives for Windows compatibility."""
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

# Configure logging with UTF-8 encoding for Windows compatibility
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = safe_log_message(msg)  # Use our emoji replacement function
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        SafeStreamHandler(sys.stdout),
        logging.FileHandler('llamaparse_pipeline.log', encoding='utf-8')
    ]
)
class SafeLogger:
    """Logger wrapper that automatically handles emoji conversion for Windows compatibility."""
    def __init__(self, logger):
        self._logger = logger
    
    def info(self, message):
        self._logger.info(safe_log_message(str(message)))
    
    def warning(self, message):
        self._logger.warning(safe_log_message(str(message)))
    
    def error(self, message):
        self._logger.error(safe_log_message(str(message)))
    
    def debug(self, message):
        self._logger.debug(safe_log_message(str(message)))

logger = SafeLogger(logging.getLogger(__name__))

try:
    # LlamaParse imports
    from llama_cloud_services import LlamaParse
    
    # LangChain imports
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    logger.info("[OK] All required libraries imported successfully")
    
except ImportError as e:
    logger.error(f"[ERR] Missing required dependencies: {e}")
    logger.error("Please install required packages:")
    logger.error("pip install llama-cloud-services langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers")
    sys.exit(1)


class LlamaParseIngestionPipeline:
    """
    A complete document ingestion pipeline using LlamaParse.
    
    This class handles the entire process from PDF discovery to vector store creation,
    providing a production-ready solution for document processing and RAG preparation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        docs_folder: str = "docs",
        base_url: Optional[str] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        num_workers: int = 3,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the LlamaParse ingestion pipeline.
        
        Args:
            api_key: LlamaParse API key (if not provided, reads from environment)
            docs_folder: Folder containing PDF documents to process
            base_url: Optional base URL for different regions (EU: https://api.cloud.eu.llamaindex.ai)
            chunk_size: Maximum size of document chunks in characters
            chunk_overlap: Overlap between chunks in characters
            num_workers: Number of parallel workers for LlamaParse processing
            embedding_model: HuggingFace model for generating embeddings
        """
        # API Configuration
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("❌ LLAMA_CLOUD_API_KEY not provided or found in environment")
        
        # Set environment variables
        os.environ["LLAMA_CLOUD_API_KEY"] = self.api_key
        if base_url:
            os.environ["LLAMA_CLOUD_BASE_URL"] = base_url
            
        # Pipeline configuration
        self.docs_folder = docs_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_workers = num_workers
        self.embedding_model = embedding_model
        
        # Processing state
        self.pdf_files: List[str] = []
        self.parsed_results: List[Dict[str, Any]] = []
        self.failed_files: List[tuple] = []
        self.all_documents: List[Document] = []
        self.chunked_documents: List[Document] = []
        self.vectorstore = None
        self.embeddings = None
        
        # Statistics
        self.processing_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_pages': 0,
            'processing_time': 0
        }
        
        logger.info("🚀 LlamaParse Document Ingestion Pipeline Initialized")
        logger.info(f"📁 Documents folder: {self.docs_folder}")
        logger.info(f"🔧 Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        logger.info(f"👥 Workers: {self.num_workers}")
        logger.info(f"🤖 Embedding model: {self.embedding_model}")
        
    def discover_documents(self) -> List[str]:
        """
        Discover PDF documents in the specified folder.
        
        Returns:
            List of PDF file paths found in the documents folder
        """
        logger.info(f"🔍 Scanning '{self.docs_folder}' folder for PDF documents...")
        
        self.pdf_files = []
        
        if not os.path.exists(self.docs_folder):
            logger.warning(f"⚠️ Warning: '{self.docs_folder}' folder not found!")
            logger.info("Please ensure your PDF documents are in the specified folder.")
            return self.pdf_files
        
        for file in os.listdir(self.docs_folder):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.docs_folder, file)
                self.pdf_files.append(pdf_path)
        
        logger.info(f"✅ Found {len(self.pdf_files)} PDF files:")
        for i, pdf_file in enumerate(self.pdf_files, 1):
            file_size = os.path.getsize(pdf_file) / 1024  # Size in KB
            logger.info(f"  {i}. {os.path.basename(pdf_file)} ({file_size:.1f} KB)")
            
        self.processing_stats['total_files'] = len(self.pdf_files)
        return self.pdf_files
    
    def process_documents(self) -> bool:
        """
        Process all discovered PDF documents using LlamaParse.
        
        Returns:
            True if processing completed successfully (even with some failures)
        """
        if not self.pdf_files:
            logger.error("❌ No PDF files to process. Run discover_documents() first.")
            return False
        
        # Initialize LlamaParse parser
        logger.info("🔧 Initializing LlamaParse parser...")
        parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            num_workers=self.num_workers,
            verbose=True,
            language="en",
            system_prompt="Focus on extracting structured data, tables, and key financial information. Preserve document hierarchy and formatting."
        )
        logger.info("✅ LlamaParse parser initialized")
        
        logger.info("🚀 Starting LlamaParse document processing...")
        logger.info("=" * 60)
        
        # Reset processing state
        self.parsed_results = []
        self.failed_files = []
        
        start_time = datetime.now()
        
        # Process each PDF file
        for i, pdf_file in enumerate(self.pdf_files, 1):
            logger.info(f"\n📄 Processing file {i}/{len(self.pdf_files)}: {os.path.basename(pdf_file)}")
            logger.info("-" * 40)
            
            try:
                logger.info("  🔄 Sending to LlamaParse API...")
                result = parser.parse(pdf_file)
                
                if result and result.pages:
                    logger.info(f"  ✅ Successfully parsed!")
                    logger.info(f"  📊 Pages processed: {len(result.pages)}")
                    
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
                    if result.pages[0].md:
                        sample_content = result.pages[0].md[:200].strip()
                        logger.info(f"  📝 Sample content: {sample_content}...")
                        
                else:
                    logger.warning(f"  ⚠️ Warning: No content extracted from {pdf_file}")
                    self.failed_files.append((pdf_file, "No content extracted"))
                    self.processing_stats['failed'] += 1
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"  ❌ Error parsing {pdf_file}: {error_msg}")
                self.failed_files.append((pdf_file, error_msg))
                self.processing_stats['failed'] += 1
        
        end_time = datetime.now()
        self.processing_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Display processing summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 LLAMAPARSE PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully processed: {self.processing_stats['successful']} files")
        logger.info(f"❌ Failed to process: {self.processing_stats['failed']} files")
        logger.info(f"📄 Total pages extracted: {self.processing_stats['total_pages']}")
        logger.info(f"⏱️ Total processing time: {self.processing_stats['processing_time']:.2f} seconds")
        
        if self.failed_files:
            logger.warning("\n❌ Failed files:")
            for file, error in self.failed_files:
                logger.warning(f"  - {os.path.basename(file)}: {error}")
        
        if self.processing_stats['successful'] > 0:
            avg_time = self.processing_stats['processing_time'] / self.processing_stats['successful']
            logger.info(f"📈 Average time per file: {avg_time:.2f} seconds")
        
        logger.info("🎉 LlamaParse processing complete!")
        return True
    
    def convert_to_documents(self) -> List[Document]:
        """
        Convert LlamaParse results to LangChain Document objects.
        
        Returns:
            List of LangChain Document objects with rich metadata
        """
        if not self.parsed_results:
            logger.error("❌ No parsed results available. Run process_documents() first.")
            return []
        
        logger.info("🔄 Converting LlamaParse results to LangChain Document format...")
        logger.info("=" * 60)
        
        self.all_documents = []
        document_stats = {
            'total_documents': 0,
            'total_content_length': 0,
            'files_processed': 0
        }
        
        for parsed_result in self.parsed_results:
            file_name = parsed_result['file_name']
            file_path = parsed_result['file_path']
            result = parsed_result['result']
            
            logger.info(f"\n📄 Converting {file_name}...")
            
            # Process each page from LlamaParse result
            for page_num, page in enumerate(result.pages, 1):
                # Use markdown content if available, fallback to text
                content = page.md if page.md else page.text
                
                if content and content.strip():
                    # Create rich metadata for each document
                    metadata = {
                        'source_file': file_path,
                        'file_name': file_name,
                        'page': page_num,
                        'total_pages': len(result.pages),
                        'parser': 'LlamaParse',
                        'result_type': 'markdown',
                        'processed_at': parsed_result['processed_at'].isoformat(),
                        'content_length': len(content)
                    }
                    
                    # Add optional metadata if available
                    if hasattr(page, 'layout') and page.layout:
                        metadata['has_layout'] = True
                    if hasattr(page, 'structuredData') and page.structuredData:
                        metadata['has_structured_data'] = True
                    if hasattr(page, 'images') and page.images:
                        metadata['image_count'] = len(page.images)
                        metadata['has_images'] = True
                    
                    # Create LangChain Document
                    document = Document(page_content=content, metadata=metadata)
                    self.all_documents.append(document)
                    document_stats['total_content_length'] += len(content)
            
            document_stats['files_processed'] += 1
            logger.info(f"  ✅ Converted {len(result.pages)} pages to LangChain Documents")
        
        document_stats['total_documents'] = len(self.all_documents)
        
        # Display conversion summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 DOCUMENT CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📁 Files processed: {document_stats['files_processed']}")
        logger.info(f"📄 Total documents created: {document_stats['total_documents']}")
        logger.info(f"📊 Total content length: {document_stats['total_content_length']:,} characters")
        
        if document_stats['total_documents'] > 0:
            avg_length = document_stats['total_content_length'] / document_stats['total_documents']
            logger.info(f"📈 Average document length: {avg_length:.0f} characters")
        
        logger.info("🎉 Document conversion complete!")
        return self.all_documents
    
    def chunk_documents(self) -> List[Document]:
        """
        Split documents into chunks optimized for RAG performance.
        
        Returns:
            List of chunked Document objects
        """
        if not self.all_documents:
            logger.error("❌ No documents available. Run convert_to_documents() first.")
            return []
        
        logger.info("🔄 Splitting documents into chunks for RAG optimization...")
        logger.info("=" * 60)
        
        # Initialize text splitter with optimized settings for LlamaParse content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n## ",      # Split on markdown headers first
                "\n### ",     # Then subheaders
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ".",         # Sentences
                " ",         # Words
                ""           # Characters
            ],
            keep_separator=True  # Keep separators to maintain markdown structure
        )
        
        logger.info("🔧 Text Splitter Configuration:")
        logger.info(f"  📏 Chunk size: {self.chunk_size} characters")
        logger.info(f"  🔗 Chunk overlap: {self.chunk_overlap} characters")
        logger.info("  📊 Hierarchical splitting: Headers → Paragraphs → Sentences")
        
        # Split all documents
        logger.info(f"\n🔄 Splitting {len(self.all_documents)} documents into chunks...")
        self.chunked_documents = text_splitter.split_documents(self.all_documents)
        
        # Analyze chunking results
        chunk_sizes = [len(doc.page_content) for doc in self.chunked_documents]
        chunk_stats = {
            'original_documents': len(self.all_documents),
            'chunked_documents': len(self.chunked_documents),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'total_content': sum(chunk_sizes)
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 DOCUMENT CHUNKING ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"📄 Original documents: {chunk_stats['original_documents']}")
        logger.info(f"🧩 Generated chunks: {chunk_stats['chunked_documents']}")
        logger.info(f"📏 Average chunk size: {chunk_stats['avg_chunk_size']:.0f} characters")
        logger.info(f"📉 Minimum chunk size: {chunk_stats['min_chunk_size']} characters")
        logger.info(f"📈 Maximum chunk size: {chunk_stats['max_chunk_size']} characters")
        
        if chunk_stats['original_documents'] > 0:
            expansion_ratio = chunk_stats['chunked_documents'] / chunk_stats['original_documents']
            logger.info(f"📈 Chunk expansion ratio: {expansion_ratio:.1f}x")
        
        # Show distribution
        size_ranges = {
            'Small (0-500)': len([s for s in chunk_sizes if s <= 500]),
            'Medium (501-1000)': len([s for s in chunk_sizes if 501 <= s <= 1000]),
            'Large (1001-1500)': len([s for s in chunk_sizes if 1001 <= s <= 1500]),
            'Extra Large (1500+)': len([s for s in chunk_sizes if s > 1500])
        }
        
        logger.info("\n📊 Chunk size distribution:")
        for range_name, count in size_ranges.items():
            percentage = (count / len(chunk_sizes) * 100) if chunk_sizes else 0
            logger.info(f"  {range_name}: {count} chunks ({percentage:.1f}%)")
        
        logger.info("✅ Document chunking complete!")
        return self.chunked_documents
    
    def create_vector_store(self) -> bool:
        """
        Create embeddings and build FAISS vector store.
        
        Returns:
            True if vector store was created successfully
        """
        if not self.chunked_documents:
            logger.error("❌ No chunked documents available. Run chunk_documents() first.")
            return False
        
        logger.info("🔄 Creating embeddings and building vector store...")
        logger.info("=" * 60)
        
        # Initialize HuggingFace embeddings
        logger.info(f"🤖 Initializing embedding model: {self.embedding_model}")
        logger.info("   Model characteristics:")
        logger.info("   - Optimized for semantic search and clustering")
        logger.info("   - 384-dimensional embeddings")
        logger.info("   - Fast inference on CPU")
        logger.info("   - Excellent performance on financial/business documents")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info("✅ Embedding model loaded successfully")
            logger.info("📊 Embedding dimension: 384")
            logger.info("💻 Device: CPU")
            logger.info("🎯 Normalization: Enabled")
            
        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            logger.error("Ensure sentence-transformers is installed: pip install sentence-transformers")
            return False
        
        # Create vector store
        logger.info(f"\n🔄 Generating embeddings for {len(self.chunked_documents)} document chunks...")
        embedding_start_time = datetime.now()
        
        try:
            self.vectorstore = FAISS.from_documents(self.chunked_documents, self.embeddings)
            
            embedding_end_time = datetime.now()
            embedding_duration = (embedding_end_time - embedding_start_time).total_seconds()
            
            logger.info("🎉 Vector store created successfully!")
            logger.info(f"⏱️ Embedding generation time: {embedding_duration:.2f} seconds")
            logger.info(f"📊 Average time per chunk: {embedding_duration/len(self.chunked_documents):.3f} seconds")
            logger.info(f"🗂️ Vector store contains {len(self.chunked_documents)} embedded chunks")
            
            # Calculate statistics
            memory_usage = len(self.chunked_documents) * 384 * 4 / 1024 / 1024
            logger.info(f"\n📈 Vector Store Statistics:")
            logger.info(f"  - Total vectors: {len(self.chunked_documents)}")
            logger.info(f"  - Embedding dimension: 384")
            logger.info(f"  - Index type: FAISS")
            logger.info(f"  - Memory usage: ~{memory_usage:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {e}")
            logger.error("Check available memory and dependencies")
            return False
    
    def test_retrieval(self, queries: Optional[List[str]] = None) -> bool:
        """
        Test the document retrieval system with sample queries.
        
        Args:
            queries: Optional list of test queries. If not provided, uses default financial queries.
            
        Returns:
            True if retrieval testing completed successfully
        """
        if not self.vectorstore:
            logger.error("❌ No vector store available. Run create_vector_store() first.")
            return False
        
        if queries is None:
            queries = [
                "investment fund regulations and compliance requirements",
                "trading restrictions and market access rules",
                "risk management framework and procedures",
                "financial reporting standards and disclosure"
            ]
        
        logger.info("🔍 Testing document retrieval system...")
        logger.info("=" * 60)
        logger.info(f"📄 Searching across {len(self.chunked_documents)} document chunks")
        logger.info(f"🎯 Testing {len(queries)} queries")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\n🔍 Query {i}/{len(queries)}: '{query}'")
            logger.info("-" * 40)
            
            try:
                # Perform similarity search
                relevant_docs = self.vectorstore.similarity_search(query, k=3)
                
                if relevant_docs:
                    for j, doc in enumerate(relevant_docs, 1):
                        source_file = doc.metadata.get('file_name', 'Unknown')
                        page_num = doc.metadata.get('page', 'Unknown')
                        content_length = len(doc.page_content)
                        
                        logger.info(f"  📋 Result {j}:")
                        logger.info(f"    📄 Source: {source_file} (Page {page_num})")
                        logger.info(f"    📏 Length: {content_length} chars")
                        
                        # Show content preview
                        preview = doc.page_content[:150].strip()
                        preview = preview.replace('#', '').replace('**', '').replace('*', '')
                        logger.info(f"    📝 Preview: {preview}...")
                else:
                    logger.warning(f"  ⚠️ No relevant documents found")
                    
            except Exception as e:
                logger.error(f"  ❌ Error during search: {e}")
        
        logger.info("🎉 Document retrieval testing complete!")
        return True
    
    def save_pipeline_results(self, output_dir: str = "output") -> Dict[str, str]:
        """
        Save all pipeline results and configuration to disk.
        
        Args:
            output_dir: Directory to save results (will be created if it doesn't exist)
            
        Returns:
            Dictionary mapping result type to file path
        """
        if not self.vectorstore:
            logger.error("❌ No results to save. Run the complete pipeline first.")
            return {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("💾 Saving pipeline results...")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"llamaparse_{timestamp}"
        saved_files = {}
        
        try:
            # Save vector store
            vector_store_path = os.path.join(output_dir, "vector_store_llamaparse")
            self.vectorstore.save_local(vector_store_path)
            saved_files['vector_store'] = vector_store_path
            logger.info(f"✅ Vector store saved to '{vector_store_path}'")
            
            # Save processed documents
            documents_file = os.path.join(output_dir, f"processed_documents_{timestamp}.pkl")
            with open(documents_file, "wb") as f:
                pickle.dump(self.chunked_documents, f)
            saved_files['documents'] = documents_file
            logger.info(f"✅ Processed documents saved to '{documents_file}'")
            
            # Save configuration
            config = {
                'session_id': session_id,
                'timestamp': timestamp,
                'pipeline_config': {
                    'docs_folder': self.docs_folder,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'num_workers': self.num_workers,
                    'embedding_model': self.embedding_model
                },
                'processing_stats': self.processing_stats,
                'files_processed': [r['file_name'] for r in self.parsed_results],
                'failed_files': [(os.path.basename(f), e) for f, e in self.failed_files]
            }
            
            config_file = os.path.join(output_dir, f"pipeline_config_{timestamp}.json")
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2, default=str)
            saved_files['config'] = config_file
            logger.info(f"✅ Configuration saved to '{config_file}'")
            
            # Save raw results
            results_file = os.path.join(output_dir, f"raw_results_{timestamp}.pkl")
            with open(results_file, "wb") as f:
                pickle.dump(self.parsed_results, f)
            saved_files['raw_results'] = results_file
            logger.info(f"✅ Raw results saved to '{results_file}'")
            
            # Create summary report
            report_file = os.path.join(output_dir, f"processing_report_{timestamp}.md")
            with open(report_file, "w") as f:
                f.write(f"# LlamaParse Processing Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## Summary\n")
                f.write(f"- **Session ID**: {session_id}\n")
                f.write(f"- **Files Processed**: {self.processing_stats['successful']}/{self.processing_stats['total_files']}\n")
                f.write(f"- **Total Pages**: {self.processing_stats['total_pages']}\n")
                f.write(f"- **Documents Created**: {len(self.all_documents)}\n")
                f.write(f"- **Chunks Generated**: {len(self.chunked_documents)}\n")
                f.write(f"- **Processing Time**: {self.processing_stats['processing_time']:.2f} seconds\n\n")
                
                if self.parsed_results:
                    f.write(f"## Successfully Processed Files\n")
                    for result in self.parsed_results:
                        f.write(f"- {result['file_name']} ({result['pages']} pages)\n")
                
                if self.failed_files:
                    f.write(f"\n## Failed Files\n")
                    for file, error in self.failed_files:
                        f.write(f"- {os.path.basename(file)}: {error}\n")
                
                f.write(f"\n## Configuration\n")
                f.write(f"- **Parser**: LlamaParse (markdown output)\n")
                f.write(f"- **Chunking**: {len(self.chunked_documents)} chunks\n")
                f.write(f"- **Embeddings**: {self.embedding_model}\n")
                f.write(f"- **Vector Store**: FAISS\n")
            
            saved_files['report'] = report_file
            logger.info(f"✅ Report saved to '{report_file}'")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
            return {}
        
        logger.info("\n🎉 All pipeline results saved successfully!")
        return saved_files
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete ingestion pipeline from start to finish.
        
        Returns:
            True if pipeline completed successfully
        """
        logger.info("🚀 Starting Complete LlamaParse Ingestion Pipeline")
        logger.info("=" * 70)
        
        try:
            # Step 1: Discover documents
            if not self.discover_documents():
                logger.error("❌ No documents found to process")
                return False
            
            # Step 2: Process documents with LlamaParse
            if not self.process_documents():
                logger.error("❌ Document processing failed")
                return False
            
            # Step 3: Convert to LangChain documents
            if not self.convert_to_documents():
                logger.error("❌ Document conversion failed")
                return False
            
            # Step 4: Chunk documents
            if not self.chunk_documents():
                logger.error("❌ Document chunking failed")
                return False
            
            # Step 5: Create vector store
            if not self.create_vector_store():
                logger.error("❌ Vector store creation failed")
                return False
            
            # Step 6: Test retrieval
            self.test_retrieval()
            
            # Step 7: Save results
            saved_files = self.save_pipeline_results()
            
            # Final summary
            logger.info("\n" + "=" * 70)
            logger.info("🎉 LLAMAPARSE INGESTION PIPELINE COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"✅ Successfully processed {self.processing_stats['successful']} PDF files")
            logger.info(f"✅ Generated {self.processing_stats['total_pages']} pages of structured content")
            logger.info(f"✅ Created {len(self.all_documents)} LangChain documents")
            logger.info(f"✅ Split into {len(self.chunked_documents)} optimized chunks")
            logger.info(f"✅ Built vector store with 384-dimensional embeddings")
            logger.info(f"✅ Total processing time: {self.processing_stats['processing_time']:.2f} seconds")
            
            if saved_files:
                logger.info(f"\n📂 Output Files:")
                for result_type, file_path in saved_files.items():
                    logger.info(f"  - {result_type}: {file_path}")
            
            logger.info(f"\n🚀 Ready for RAG Implementation!")
            logger.info(f"Your vector store is ready for use in RAG pipelines.")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {e}")
            return False


def main():
    """
    Main function to run the LlamaParse ingestion pipeline.
    """
    # Configuration - you can modify these values or make them command-line arguments
    API_KEY = "llx-MHXNHhQO6ahnPlDx7i5O3QYgdYdaVmhtWIFyrJPI1zszoSu8"
    BASE_URL = "https://api.cloud.eu.llamaindex.ai"  # EU region
    DOCS_FOLDER = "docs"
    
    try:
        # Initialize pipeline
        pipeline = LlamaParseIngestionPipeline(
            api_key=API_KEY,
            docs_folder=DOCS_FOLDER,
            base_url=BASE_URL,
            chunk_size=1500,
            chunk_overlap=300,
            num_workers=1  # Single worker for stability
        )
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()