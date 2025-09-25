# LlamaParse Document Ingestion Pipeline

A **modular, production-ready document ingestion pipeline** using LlamaParse that transforms PDF documents into searchable vector stores for RAG (Retrieval-Augmented Generation) applications.

## 🚀 Key Improvements (Modular Architecture)

The pipeline has been **completely refactored** from a single 800+ line monolithic script into a clean, modular architecture following Python best practices:

### ✅ **Benefits of the New Architecture**

- **🔧 Modular Design**: Each component has a single responsibility
- **🧪 Testable**: Individual components can be unit tested
- **🔄 Reusable**: Components can be used independently in other projects
- **📦 Maintainable**: Clear separation of concerns, easier debugging
- **⚡ Type-Safe**: Full type hints and dataclass configurations
- **🔌 Extensible**: Easy to add new processors or stores

## 📁 New Modular Project Structure

```
llamaparse_pipeline/
├── __init__.py                      # Main package exports
├── main.py                          # Clean entry point (< 50 lines)
├── example_step_by_step.py          # Usage examples
├── llamaparse_ingestion_pipeline.py # Original script (kept for reference)
├── requirements.txt                 # Python dependencies
├── config/
│   ├── __init__.py
│   └── settings.py                  # Centralized configuration
├── utils/
│   ├── __init__.py
│   ├── logger.py                    # Windows-compatible logging
│   └── helpers.py                   # Common utilities
├── processors/
│   ├── __init__.py
│   ├── llamaparse_processor.py      # LlamaParse API handling
│   └── document_converter.py        # Document conversion & chunking
├── stores/
│   ├── __init__.py
│   └── vector_store.py              # Embedding & vector store management
├── pipeline/
│   ├── __init__.py
│   └── ingestion_pipeline.py        # Main orchestrator
├── docs/                            # PDF documents to process
├── notebooks/                       # Original Jupyter notebooks (archived)
└── output/                          # Generated results (created automatically)
```

## 🏗️ Architecture Components

### 1. **Configuration (`config/`)**
- `PipelineConfig`: Main pipeline configuration with type safety
- `LlamaParseConfig`: API and processing settings
- `ChunkingConfig`: Document splitting parameters  
- `EmbeddingConfig`: Embedding model settings

### 2. **Processors (`processors/`)**
- `LlamaParseProcessor`: Handles LlamaParse API interactions and document discovery
- `DocumentConverter`: Converts results to LangChain format and performs intelligent chunking

### 3. **Stores (`stores/`)**
- `VectorStoreManager`: Manages embeddings and FAISS vector store operations

### 4. **Pipeline (`pipeline/`)**
- `IngestionPipeline`: Orchestrates all components with proper error handling

### 5. **Utils (`utils/`)**
- `logger.py`: Windows-compatible logging with emoji conversion
- `helpers.py`: File operations, statistics, and reporting utilities

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `pipeline_config.ini` or set environment variable:

```bash
# Option 1: Environment variable
export LLAMA_CLOUD_API_KEY="your-api-key-here"

# Option 2: Edit pipeline_config.ini
# LLAMA_CLOUD_API_KEY = your-api-key-here
```

### 3. Add Documents

Place your PDF documents in the `docs/` folder:

```bash
mkdir -p docs
# Copy your PDF files to docs/
```

## 🚀 Usage

### Simple Usage (Recommended)
```python
from pipeline import IngestionPipeline
from config import PipelineConfig

# Create configuration
config = PipelineConfig.create_default(
    api_key="your-llamaparse-api-key",
    docs_folder="docs"
)

# Run complete pipeline
pipeline = IngestionPipeline(config)
success = pipeline.run_complete_pipeline()
```

### Command Line (Easiest)
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key in main.py or as environment variable
export LLAMA_CLOUD_API_KEY=your_api_key_here

# Place PDFs in docs/ folder and run
python main.py
```

### Advanced Usage (Step-by-Step Control)
```python
from config.settings import PipelineConfig
from processors.llamaparse_processor import LlamaParseProcessor
from processors.document_converter import DocumentConverter
from stores.vector_store import VectorStoreManager

# Initialize components
config = PipelineConfig.create_default(api_key="your-key")
processor = LlamaParseProcessor(config.llamaparse)
converter = DocumentConverter(config.chunking)
vector_manager = VectorStoreManager(config.embedding)

# Process step by step
pdf_files = processor.discover_documents("docs")
processor.process_documents()
documents = converter.convert_to_documents(processor.get_results())
chunks = converter.chunk_documents(documents)
vector_manager.create_vector_store(chunks)

# Search
results = vector_manager.similarity_search("your query", k=3)
```

### Legacy Usage (Original Script)
```bash
# Still available for reference
python llamaparse_ingestion_pipeline.py
```

## 📊 Pipeline Steps

1. **Document Discovery**: Scans the `docs/` folder for PDF files
2. **LlamaParse Processing**: Uses LlamaParse API to extract structured content
3. **Document Conversion**: Transforms results to LangChain Document objects
4. **Text Chunking**: Splits documents into optimized chunks for RAG
5. **Vectorization**: Creates embeddings using HuggingFace models
6. **Vector Store Creation**: Builds FAISS vector store for fast retrieval
7. **Testing**: Validates retrieval with sample queries
8. **Persistence**: Saves all results and configuration

## 📂 Output Files

The pipeline generates several output files in the `output/` directory:

- `vector_store_llamaparse/`: FAISS vector store for RAG applications
- `processed_documents_TIMESTAMP.pkl`: Chunked documents with metadata
- `pipeline_config_TIMESTAMP.json`: Complete pipeline configuration
- `raw_results_TIMESTAMP.pkl`: Original LlamaParse results
- `processing_report_TIMESTAMP.md`: Human-readable processing summary

## ⚙️ Configuration Options

### Pipeline Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `docs_folder` | `"docs"` | Folder containing PDF documents |
| `chunk_size` | `1500` | Maximum chunk size in characters |
| `chunk_overlap` | `300` | Overlap between chunks |
| `num_workers` | `1` | LlamaParse parallel workers |
| `embedding_model` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace embedding model |

### API Configuration

| Setting | Description |
|---------|-------------|
| `LLAMA_CLOUD_API_KEY` | Your LlamaParse API key |
| `LLAMA_CLOUD_BASE_URL` | API endpoint (use EU region if needed) |

## 🌍 Regional Configuration

For EU users, set the base URL to the EU region:

```python
pipeline = LlamaParseIngestionPipeline(
    base_url="https://api.cloud.eu.llamaindex.ai"
)
```

## 📝 Logging

The pipeline provides comprehensive logging:

- Console output with colored formatting
- Log file: `llamaparse_pipeline.log`
- Multiple log levels: INFO, WARNING, ERROR

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Install dependencies with `pip install -r requirements.txt`
2. **API Key Issues**: Verify your API key and region settings
3. **Memory Issues**: Reduce `chunk_size` or process fewer documents
4. **No Documents Found**: Check that PDFs are in the `docs/` folder

### Debug Mode

Enable verbose logging by modifying the logging level:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 🚀 Using Results in RAG Applications

After running the pipeline, use the generated vector store:

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load the saved vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local("output/vector_store_llamaparse", embeddings)

# Use for retrieval
results = vectorstore.similarity_search("your query", k=5)
```

## 📈 Performance Tips

1. **Use EU Region**: If in Europe, set base URL to EU region for faster processing
2. **Single Worker**: Use `num_workers=1` for stability with complex documents
3. **Optimize Chunks**: Adjust `chunk_size` based on your document types
4. **GPU Acceleration**: Install PyTorch with CUDA for faster embeddings

## 🤝 Integration with Other Tools

The pipeline outputs are compatible with:

- **LangChain**: Use generated documents and vector stores directly
- **LlamaIndex**: Convert documents to LlamaIndex format
- **OpenAI**: Use with GPT models for question answering
- **Hugging Face**: Compatible with all Hugging Face embedding models

## 📚 Additional Resources

- [LlamaParse Documentation](https://docs.cloud.llamaindex.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## 📄 License

This project is provided as-is for educational and commercial use. Please respect the terms of service for LlamaParse and other integrated services.