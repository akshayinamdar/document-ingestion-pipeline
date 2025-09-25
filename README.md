# LlamaParse Document Ingestion Pipeline

A production-ready Python script for processing PDF documents using **LlamaParse** and preparing them for RAG (Retrieval-Augmented Generation) applications.

## 🚀 Features

- **No Jupyter Dependencies**: Pure Python script eliminates async/event loop issues
- **Production Ready**: Comprehensive error handling, logging, and configuration
- **Flexible Configuration**: Easy customization through configuration files or environment variables
- **Complete Pipeline**: From PDF discovery to vector store creation
- **Rich Metadata**: Detailed document metadata for enhanced retrieval
- **Multiple Output Formats**: Vector stores, processed documents, configuration files, and reports

## 📁 Project Structure

```
Deutsche Börse/
├── llamaparse_ingestion_pipeline.py  # Main pipeline script
├── requirements.txt                   # Python dependencies
├── pipeline_config.ini               # Configuration file
├── docs/                             # PDF documents to process
├── notebooks/                        # Original Jupyter notebooks (archived)
│   ├── 01-Ingestion-Pipeline.ipynb
│   ├── 02-Docling-RAG-Pipeline.ipynb
│   └── 03-Ingestion-Pipeline-LlamaParse.ipynb
└── output/                           # Generated results (created automatically)
```

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

### Basic Usage

```bash
python llamaparse_ingestion_pipeline.py
```

### Advanced Usage

You can customize the pipeline by modifying the configuration in the script or using environment variables:

```python
from llamaparse_ingestion_pipeline import LlamaParseIngestionPipeline

# Initialize with custom settings
pipeline = LlamaParseIngestionPipeline(
    api_key="your-api-key",
    docs_folder="custom_docs",
    base_url="https://api.cloud.eu.llamaindex.ai",  # For EU region
    chunk_size=2000,
    chunk_overlap=400,
    num_workers=2
)

# Run individual steps
pipeline.discover_documents()
pipeline.process_documents()
pipeline.convert_to_documents()
pipeline.chunk_documents()
pipeline.create_vector_store()
pipeline.test_retrieval()
pipeline.save_pipeline_results()

# Or run everything at once
success = pipeline.run_complete_pipeline()
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