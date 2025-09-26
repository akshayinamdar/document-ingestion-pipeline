"""
Document converter and chunking processor.

This module handles conversion of LlamaParse results to LangChain documents
and intelligent document chunking for RAG optimization.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.logger import get_logger
from utils.helpers import calculate_statistics
from config.settings import ChunkingConfig

# LlamaIndex imports for semantic chunking
try:
    from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.schema import Document as LlamaDocument, TextNode
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    # Create dummy classes to avoid NameError when LlamaIndex is not available
    class LlamaDocument:
        def __init__(self, text="", metadata=None, doc_id=""):
            self.text = text
            self.metadata = metadata or {}
            self.doc_id = doc_id
    
    class TextNode:
        def __init__(self, text="", metadata=None, node_id=""):
            self.text = text
            self.metadata = metadata or {}
            self.node_id = node_id
    
    class SemanticSplitterNodeParser:
        pass
    
    class HuggingFaceEmbedding:
        pass
    
    LLAMAINDEX_AVAILABLE = False


class DocumentConverter:
    """
    Converts LlamaParse results to LangChain Documents and handles chunking.
    
    This class manages the transformation from raw LlamaParse results to
    structured LangChain Document objects with rich metadata.
    """
    
    def __init__(self, chunking_config: ChunkingConfig):
        """
        Initialize the document converter.
        
        Args:
            chunking_config: Configuration for document chunking
        """
        self.config = chunking_config
        self.logger = get_logger(__name__)
        
        # Processing state
        self.all_documents: List[Document] = []
        self.chunked_documents: List[Document] = []
        
        self.logger.info("[CONFIG] Document converter initialized")
        self.logger.info(f"[CONFIG] Chunk size: {chunking_config.chunk_size}")
        self.logger.info(f"[CONFIG] Chunk overlap: {chunking_config.chunk_overlap}")
    
    def convert_to_documents(self, parsed_results: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert LlamaParse results to LangChain Document objects.
        
        Args:
            parsed_results: List of parsed results from LlamaParse
            
        Returns:
            List of LangChain Document objects with rich metadata
        """
        if not parsed_results:
            self.logger.error("[ERR] No parsed results available for conversion.")
            return []
        
        self.logger.info("[PROC] Converting LlamaParse results to LangChain Document format...")
        self.logger.info("=" * 60)
        
        self.all_documents = []
        document_stats = {
            'total_documents': 0,
            'total_content_length': 0,
            'files_processed': 0
        }
        
        for parsed_result in parsed_results:
            file_name = parsed_result['file_name']
            file_path = parsed_result['file_path']
            result = parsed_result['result']
            
            self.logger.info(f"\n[DOC] Converting {file_name}...")
            
            # Process each page from LlamaParse result
            page_documents = []
            for page_num, page in enumerate(result.pages, 1):
                # Use markdown content if available, fallback to text
                content = page.md if hasattr(page, 'md') and page.md else page.text
                
                if content and content.strip():
                    # Create rich metadata for each document
                    metadata = self._create_document_metadata(
                        file_path, file_name, page_num, len(result.pages),
                        parsed_result['processed_at'], content, page
                    )
                    
                    # Create LangChain Document
                    document = Document(page_content=content, metadata=metadata)
                    page_documents.append(document)
                    document_stats['total_content_length'] += len(content)
            
            self.all_documents.extend(page_documents)
            document_stats['files_processed'] += 1
            
            self.logger.info(f"  [OK] Converted {len(page_documents)} pages to LangChain Documents")
        
        document_stats['total_documents'] = len(self.all_documents)
        
        # Display conversion summary
        self._log_conversion_summary(document_stats)
        
        return self.all_documents
    
    def _create_document_metadata(
        self,
        file_path: str,
        file_name: str,
        page_num: int,
        total_pages: int,
        processed_at: datetime,
        content: str,
        page: Any
    ) -> Dict[str, Any]:
        """
        Create rich metadata for a document.
        
        Args:
            file_path: Full path to source file
            file_name: Name of source file
            page_num: Page number
            total_pages: Total pages in document
            processed_at: When the document was processed
            content: Page content
            page: LlamaParse page object
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'source_file': file_path,
            'file_name': file_name,
            'page': page_num,
            'total_pages': total_pages,
            'parser': 'LlamaParse',
            'result_type': 'markdown',
            'processed_at': processed_at.isoformat(),
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
        
        return metadata
    
    def chunk_documents(self, documents: List[Document] = None) -> List[Document]:
        """
        Split documents into chunks optimized for RAG performance.
        
        Automatically chooses between semantic and traditional chunking based on configuration.
        
        Args:
            documents: Documents to chunk (uses internal documents if None)
            
        Returns:
            List of chunked Document objects
        """
        if documents is None:
            documents = self.all_documents
        
        if not documents:
            self.logger.error("[ERR] No documents available for chunking.")
            return []
        
        # Choose chunking method based on configuration
        return self.semantic_chunk_documents(documents)
    
    def semantic_chunk_documents(self, documents: List[Document] = None) -> List[Document]:
        """
        Split documents using semantic chunking from LlamaIndex.
        
        This method uses embedding similarity to find natural breakpoints
        between sentences, creating chunks that contain semantically related content.
        
        Args:
            documents: Documents to chunk (uses internal documents if None)
            
        Returns:
            List of semantically chunked Document objects
        """
        if documents is None:
            documents = self.all_documents
        
        if not documents:
            self.logger.error("[ERR] No documents available for semantic chunking.")
            return []
        
        if not LLAMAINDEX_AVAILABLE:
            self.logger.error("[ERR] LlamaIndex is not installed. Please install llama-index-core and llama-index-embeddings-openai")
            self.logger.info("[FALLBACK] Using traditional chunking instead...")
            return self.chunk_documents(documents)
        
        self.logger.info("[PROC] Performing semantic chunking using LlamaIndex...")
        self.logger.info("=" * 60)
        
        # Initialize HuggingFace embedding model
        try:
            embed_model = HuggingFaceEmbedding(
                model_name=self.config.embed_model_name,
                device="cpu"
            )
        except Exception as e:
            self.logger.error(f"[ERR] Failed to initialize HuggingFace embeddings: {e}")
            self.logger.info("[FALLBACK] Using traditional chunking instead...")
            return self.chunk_documents(documents)
        
        # Initialize semantic splitter
        splitter = SemanticSplitterNodeParser(
            buffer_size=self.config.buffer_size,
            breakpoint_percentile_threshold=self.config.breakpoint_percentile_threshold,
            embed_model=embed_model
        )
        
        self.logger.info("[CONFIG] Semantic Splitter Configuration:")
        self.logger.info(f"  [BUFFER] Buffer size: {self.config.buffer_size}")
        self.logger.info(f"  [THRESHOLD] Breakpoint percentile: {self.config.breakpoint_percentile_threshold}%")
        self.logger.info(f"  [MODEL] Embedding model: {self.config.embed_model_name}")
        self.logger.info(f"  [DEVICE] Device: CPU")
        self.logger.info(f"  [DIMS] Embedding dimensions: 384")
        
        # Convert LangChain documents to LlamaIndex documents
        self.logger.info(f"\n[PROC] Converting {len(documents)} documents for semantic analysis...")
        llama_documents = self._convert_to_llama_documents(documents)
        
        # Apply semantic chunking
        self.logger.info("[PROC] Analyzing semantic relationships and creating chunks...")
        try:
            nodes = splitter.get_nodes_from_documents(llama_documents)
            self.logger.info(f"[SUCCESS] Generated {len(nodes)} semantic chunks")
        except Exception as e:
            self.logger.error(f"[ERR] Semantic chunking failed: {e}")
            self.logger.info("[FALLBACK] Using traditional chunking instead...")
            return self.chunk_documents(documents)
        
        # Convert back to LangChain documents
        self.chunked_documents = self._convert_nodes_to_langchain_documents(nodes, documents)
        
        # Analyze chunking results
        self._analyze_semantic_chunking_results(documents)
        
        return self.chunked_documents
    
    def _convert_to_llama_documents(self, documents: List[Document]) -> List[LlamaDocument]:
        """
        Convert LangChain documents to LlamaIndex documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of LlamaIndex Document objects
        """
        llama_docs = []
        for i, doc in enumerate(documents):
            llama_doc = LlamaDocument(
                text=doc.page_content,
                metadata=doc.metadata,
                doc_id=f"doc_{i}"
            )
            llama_docs.append(llama_doc)
        return llama_docs
    
    def _convert_nodes_to_langchain_documents(self, nodes: List[TextNode], original_documents: List[Document]) -> List[Document]:
        """
        Convert LlamaIndex nodes back to LangChain documents while preserving metadata.
        
        Args:
            nodes: List of LlamaIndex TextNode objects
            original_documents: Original LangChain documents for metadata reference
            
        Returns:
            List of LangChain Document objects
        """
        langchain_docs = []
        
        for node in nodes:
            # Create base metadata from the node
            metadata = dict(node.metadata) if node.metadata else {}
            
            # Add semantic chunking specific metadata
            metadata.update({
                'chunk_id': node.node_id,
                'chunking_method': 'semantic',
                'semantic_threshold': self.config.breakpoint_percentile_threshold,
                'buffer_size': self.config.buffer_size,
                'embedding_model': self.config.embed_model_name,
                'embedding_provider': 'huggingface'
            })
            
            # Create LangChain document
            doc = Document(
                page_content=node.text,
                metadata=metadata
            )
            langchain_docs.append(doc)
        
        return langchain_docs
    
    def _analyze_semantic_chunking_results(self, original_documents: List[Document]) -> None:
        """
        Analyze and log semantic chunking results.
        
        Args:
            original_documents: Original documents before chunking
        """
        if not self.chunked_documents:
            return
        
        chunk_sizes = [len(doc.page_content) for doc in self.chunked_documents]
        stats = calculate_statistics(chunk_sizes)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("[STATS] SEMANTIC CHUNKING ANALYSIS")
        self.logger.info("=" * 60)
        self.logger.info(f"[DOC] Original documents: {len(original_documents)}")
        self.logger.info(f"[CHUNK] Generated semantic chunks: {stats['count']}")
        self.logger.info(f"[SIZE] Average chunk size: {stats['mean']:.0f} characters")
        self.logger.info(f"[SIZE] Minimum chunk size: {stats['min']} characters")
        self.logger.info(f"[SIZE] Maximum chunk size: {stats['max']} characters")
        
        if len(original_documents) > 0:
            expansion_ratio = stats['count'] / len(original_documents)
            self.logger.info(f"[GRAPH] Chunk expansion ratio: {expansion_ratio:.1f}x")
        
        # Show size distribution
        size_ranges = {
            'Small (0-500)': len([s for s in chunk_sizes if s <= 500]),
            'Medium (501-1000)': len([s for s in chunk_sizes if 501 <= s <= 1000]),
            'Large (1001-1500)': len([s for s in chunk_sizes if 1001 <= s <= 1500]),
            'Extra Large (1500+)': len([s for s in chunk_sizes if s > 1500])
        }
        
        self.logger.info("\n[STATS] Semantic chunk size distribution:")
        for range_name, count in size_ranges.items():
            percentage = (count / len(chunk_sizes) * 100) if chunk_sizes else 0
            self.logger.info(f"  {range_name}: {count} chunks ({percentage:.1f}%)")
        
        self.logger.info("[SUCCESS] Semantic chunking complete!")
    
    def _analyze_chunking_results(self, original_documents: List[Document]) -> None:
        """
        Analyze and log chunking results.
        
        Args:
            original_documents: Original documents before chunking
        """
        if not self.chunked_documents:
            return
        
        chunk_sizes = [len(doc.page_content) for doc in self.chunked_documents]
        stats = calculate_statistics(chunk_sizes)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("[STATS] DOCUMENT CHUNKING ANALYSIS")
        self.logger.info("=" * 60)
        self.logger.info(f"[DOC] Original documents: {len(original_documents)}")
        self.logger.info(f"[CHUNK] Generated chunks: {stats['count']}")
        self.logger.info(f"[SIZE] Average chunk size: {stats['mean']:.0f} characters")
        self.logger.info(f"[SIZE] Minimum chunk size: {stats['min']} characters")
        self.logger.info(f"[SIZE] Maximum chunk size: {stats['max']} characters")
        
        if len(original_documents) > 0:
            expansion_ratio = stats['count'] / len(original_documents)
            self.logger.info(f"[GRAPH] Chunk expansion ratio: {expansion_ratio:.1f}x")
        
        # Show size distribution
        size_ranges = {
            'Small (0-500)': len([s for s in chunk_sizes if s <= 500]),
            'Medium (501-1000)': len([s for s in chunk_sizes if 501 <= s <= 1000]),
            'Large (1001-1500)': len([s for s in chunk_sizes if 1001 <= s <= 1500]),
            'Extra Large (1500+)': len([s for s in chunk_sizes if s > 1500])
        }
        
        self.logger.info("\n[STATS] Chunk size distribution:")
        for range_name, count in size_ranges.items():
            percentage = (count / len(chunk_sizes) * 100) if chunk_sizes else 0
            self.logger.info(f"  {range_name}: {count} chunks ({percentage:.1f}%)")
        
        self.logger.info("[OK] Document chunking complete!")
    
    def _log_conversion_summary(self, stats: Dict[str, Any]) -> None:
        """
        Log document conversion summary.
        
        Args:
            stats: Conversion statistics
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("[LIST] DOCUMENT CONVERSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"[FOLDER] Files processed: {stats['files_processed']}")
        self.logger.info(f"[DOC] Total documents created: {stats['total_documents']}")
        self.logger.info(f"[STATS] Total content length: {stats['total_content_length']:,} characters")
        
        if stats['total_documents'] > 0:
            avg_length = stats['total_content_length'] / stats['total_documents']
            self.logger.info(f"[GRAPH] Average document length: {avg_length:.0f} characters")
        
        self.logger.info("[SUCCESS] Document conversion complete!")
    
    def get_documents(self) -> List[Document]:
        """
        Get the converted documents.
        
        Returns:
            List of LangChain Document objects
        """
        return self.all_documents
    
    def get_chunks(self) -> List[Document]:
        """
        Get the chunked documents.
        
        Returns:
            List of chunked Document objects
        """
        return self.chunked_documents
    
    def has_documents(self) -> bool:
        """
        Check if there are any converted documents.
        
        Returns:
            True if there are documents
        """
        return len(self.all_documents) > 0
    
    def has_chunks(self) -> bool:
        """
        Check if there are any chunked documents.
        
        Returns:
            True if there are chunks
        """
        return len(self.chunked_documents) > 0