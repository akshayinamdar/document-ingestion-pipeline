# Configuration file for API credentials
import os

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')

# You can set your API key in one of these ways:
# 1. Set environment variable: OPENAI_API_KEY
# 2. Replace 'your-openai-api-key-here' with your actual key

# Usage in PowerShell to set environment variable:
# $env:OPENAI_API_KEY = "your-actual-api-key"

# Other configuration options
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200