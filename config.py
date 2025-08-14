import os
from dotenv import load_dotenv
load_dotenv() 

# Read from environment/secrets (Streamlit Cloud: st.secrets; HF Spaces: Repository Secrets)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "medical-rag")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")

# Model/Index settings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
EMBED_DIM   = 384
TOP_K       = 5

# Chunking
CHUNK_SIZE      = 900
CHUNK_OVERLAP   = 120

# Streamlit toggles
SHOW_RETRIEVED_CHUNKS = True
