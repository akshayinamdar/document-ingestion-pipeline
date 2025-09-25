"""
Vector store manager for embedding creation and similarity search.

This module handles embedding generation, FAISS vector store creation,
and provides retrieval capabilities for RAG applications.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import get_logger
from utils.helpers import format_duration
from config.settings import EmbeddingConfig, DEFAULT_TEST_QUERIES


class VectorStoreManager:
    """
    Manages embedding generation and FAISS vector store operations.
    
    This class handles the creation of embeddings from document chunks
    and provides search capabilities for retrieval-augmented generation.
    """
    
    def __init__(self, embedding_config: EmbeddingConfig):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_config: Configuration for embeddings
        """
        self.config = embedding_config
        self.logger = get_logger(__name__)
        
        # Store state
        self.embeddings = None
        self.vectorstore = None
        
        self.logger.info("[CONFIG] Vector store manager initialized")
        self.logger.info(f"[AI] Embedding model: {embedding_config.model_name}")
        self.logger.info(f"[CPU] Device: {embedding_config.device}")
    
    def create_embeddings(self) -> bool:
        """
        Initialize the embedding model.
        
        Returns:
            True if embeddings were initialized successfully
        """
        self.logger.info(f"[AI] Initializing embedding model: {self.config.model_name}")
        self.logger.info("   Model characteristics:")
        self.logger.info("   - Optimized for semantic search and clustering")
        self.logger.info("   - 384-dimensional embeddings")
        self.logger.info("   - Fast inference on CPU")
        self.logger.info("   - Excellent performance on financial/business documents")
        
        try:
            model_kwargs = {'device': self.config.device}
            encode_kwargs = {
                'normalize_embeddings': self.config.normalize_embeddings
            }
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            self.logger.info("[OK] Embedding model loaded successfully")
            self.logger.info("[STATS] Embedding dimension: 384")
            self.logger.info(f"[CPU] Device: {self.config.device}")
            self.logger.info(f"[TARGET] Normalization: {'Enabled' if self.config.normalize_embeddings else 'Disabled'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERR] Error loading embedding model: {e}")
            self.logger.error("Ensure sentence-transformers is installed: pip install sentence-transformers")
            return False
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Create FAISS vector store from document chunks.
        
        Args:
            documents: List of document chunks to embed
            
        Returns:
            True if vector store was created successfully
        """
        if not documents:
            self.logger.error("[ERR] No documents provided for vector store creation.")
            return False
        
        if not self.embeddings:
            self.logger.info("[CONFIG] Embeddings not initialized, creating them first...")
            if not self.create_embeddings():
                return False
        
        self.logger.info(f"[PROC] Generating embeddings for {len(documents)} document chunks...")
        embedding_start_time = datetime.now()
        
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            embedding_end_time = datetime.now()
            embedding_duration = (embedding_end_time - embedding_start_time).total_seconds()
            
            self.logger.info("[SUCCESS] Vector store created successfully!")
            self.logger.info(f"[TIME] Embedding generation time: {format_duration(embedding_duration)}")
            self.logger.info(f"[STATS] Average time per chunk: {embedding_duration/len(documents):.3f} seconds")
            self.logger.info(f"[FILES] Vector store contains {len(documents)} embedded chunks")
            
            # Calculate statistics
            memory_usage = len(documents) * 384 * 4 / 1024 / 1024
            self.logger.info(f"\n[GRAPH] Vector Store Statistics:")
            self.logger.info(f"  - Total vectors: {len(documents)}")
            self.logger.info(f"  - Embedding dimension: 384")
            self.logger.info(f"  - Index type: FAISS")
            self.logger.info(f"  - Estimated memory usage: ~{memory_usage:.1f} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERR] Error creating vector store: {e}")
            self.logger.error("Check available memory and dependencies")
            return False
    
    def save_vector_store(self, save_path: str) -> bool:
        """
        Save vector store to disk.
        
        Args:
            save_path: Path to save the vector store
            
        Returns:
            True if saved successfully
        """
        if not self.vectorstore:
            self.logger.error("[ERR] No vector store to save.")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            self.vectorstore.save_local(save_path)
            self.logger.info(f"[SAVE] Vector store saved to '{save_path}'")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERR] Error saving vector store: {e}")
            return False
    
    def load_vector_store(self, load_path: str) -> bool:
        """
        Load vector store from disk.
        
        Args:
            load_path: Path to load the vector store from
            
        Returns:
            True if loaded successfully
        """
        if not self.embeddings:
            if not self.create_embeddings():
                return False
        
        try:
            self.vectorstore = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info(f"[OK] Vector store loaded from '{load_path}'")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERR] Error loading vector store: {e}")
            return False
    
    def test_retrieval(
        self, 
        queries: Optional[List[str]] = None, 
        k: int = 3,
        score_threshold: Optional[float] = None
    ) -> bool:
        """
        Test the document retrieval system with sample queries.
        
        Args:
            queries: List of test queries (uses defaults if None)
            k: Number of results to return per query
            score_threshold: Optional score threshold for filtering results
            
        Returns:
            True if retrieval testing completed successfully
        """
        if not self.vectorstore:
            self.logger.error("[ERR] No vector store available for testing.")
            return False
        
        if queries is None:
            queries = DEFAULT_TEST_QUERIES
        
        # Get total number of documents in the vector store
        total_docs = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 'Unknown'
        
        self.logger.info("[SEARCH] Testing document retrieval system...")
        self.logger.info("=" * 60)
        self.logger.info(f"[DOC] Searching across {total_docs} document chunks")
        self.logger.info(f"[TARGET] Testing {len(queries)} queries")
        self.logger.info(f"[LIST] Returning top {k} results per query")
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"\n[SEARCH] Query {i}/{len(queries)}: '{query}'")
            self.logger.info("-" * 40)
            
            try:
                # Perform similarity search
                if score_threshold is not None:
                    relevant_docs = self.vectorstore.similarity_search_with_score(query, k=k)
                    # Filter by score threshold
                    relevant_docs = [(doc, score) for doc, score in relevant_docs if score <= score_threshold]
                    docs = [doc for doc, score in relevant_docs]
                else:
                    docs = self.vectorstore.similarity_search(query, k=k)
                
                if docs:
                    for j, doc in enumerate(docs, 1):
                        source_file = doc.metadata.get('file_name', 'Unknown')
                        page_num = doc.metadata.get('page', 'Unknown')
                        content_length = len(doc.page_content)
                        
                        self.logger.info(f"  [LIST] Result {j}:")
                        self.logger.info(f"    [DOC] Source: {source_file} (Page {page_num})")
                        self.logger.info(f"    [SIZE] Length: {content_length} chars")
                        
                        # Show content preview
                        preview = self._clean_content_preview(doc.page_content)
                        self.logger.info(f"    [NOTE] Preview: {preview}...")
                        
                        # Show score if available
                        if score_threshold is not None and i <= len(relevant_docs):
                            score = relevant_docs[j-1][1]
                            self.logger.info(f"    [STAR] Similarity score: {score:.3f}")
                else:
                    self.logger.warning(f"  [WARN] No relevant documents found")
                    
            except Exception as e:
                self.logger.error(f"  [ERR] Error during search: {e}")
        
        self.logger.info("[SUCCESS] Document retrieval testing complete!")
        return True
    
    def _clean_content_preview(self, content: str, max_length: int = 150) -> str:
        """
        Clean and truncate content for preview display.
        
        Args:
            content: Content to clean
            max_length: Maximum length of preview
            
        Returns:
            Cleaned and truncated content
        """
        # Remove markdown formatting
        preview = content.replace('#', '').replace('**', '').replace('*', '')
        preview = preview.replace('\n', ' ').strip()
        
        # Truncate if too long
        if len(preview) > max_length:
            preview = preview[:max_length].strip()
        
        return preview
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Optional score threshold
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Create vector store first.")
        
        if score_threshold is not None:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            # Filter by score and return documents only
            return [doc for doc, score in results if score <= score_threshold]
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def has_vector_store(self) -> bool:
        """
        Check if vector store is available.
        
        Returns:
            True if vector store is initialized
        """
        return self.vectorstore is not None
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents in the vector store
        """
        if not self.vectorstore:
            return 0
        
        return self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 0