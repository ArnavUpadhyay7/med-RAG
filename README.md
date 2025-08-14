# 🩺 Medical Literature RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system designed for medical literature analysis and clinical decision support. This system provides evidence-based answers to clinical questions using advanced embedding models, vector databases, and large language models.

## ✨ Features

- **📚 PDF Ingestion**: Upload and process medical PDFs with intelligent chunking
- **🔍 Semantic Search**: Advanced retrieval using Sentence Transformers and Pinecone
- **💡 AI Generation**: Evidence-based answers using Groq LLM with source citations
- **📊 Evaluation Suite**: Built-in retrieval accuracy and performance metrics
- **🎯 Medical Focus**: Optimized for clinical questions and medical literature
- **⚡ Performance**: Real-time retrieval and generation with latency tracking

## 🏗️ Architecture

### Core Components

1. **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
2. **Vector Database**: Pinecone for scalable similarity search
3. **Text Chunking**: Recursive character splitting optimized for medical documents
4. **LLM Integration**: Groq API with Llama3-8B model
5. **Evaluation**: Retrieval accuracy metrics and performance diagnostics

### Data Flow

```
PDF Upload → Text Extraction → Chunking → Embedding → Pinecone Index
                                                      ↓
Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Answer
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Pinecone API key
- Groq API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd medical_rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Environment Variables

Create a `.env` file with:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=medical-rag
```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage

### 1. Upload Medical PDFs
- Use the sidebar to upload one or more medical PDF files
- The system will automatically chunk and index the documents

### 2. Ask Clinical Questions
- Enter your clinical question in natural language
- Example: "What is the first-line treatment for stage 2 hypertension in patients with diabetes?"

### 3. Review Results
- **Evidence-based Answer**: AI-generated response with source citations
- **Source Documents**: Retrieved chunks with relevance scores
- **Performance Metrics**: Retrieval and generation latency

### 4. Evaluation
- Run the built-in evaluation suite to assess system performance
- View accuracy metrics and detailed results for each test question

## 🔧 Configuration

Key configuration options in `config.py`:

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
EMBED_DIM = 384                                          # Vector dimensions
CHUNK_SIZE = 900                                         # Text chunk size
CHUNK_OVERLAP = 120                                      # Chunk overlap
TOP_K = 5                                                # Retrieval count
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation including:

- **Retrieval Accuracy**: Percentage of expected keywords found in retrieved chunks
- **Latency Metrics**: Retrieval and generation response times
- **Relevance Scoring**: Cosine similarity scores for retrieved documents
- **Source Coverage**: Analysis of which sources contribute to answers

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set the following secrets:
   - `GROQ_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX`
4. Deploy with entry point: `app.py`

### Hugging Face Spaces

1. Create a new Streamlit space on [Hugging Face](https://huggingface.co/spaces)
2. Upload your code or connect your GitHub repository
3. Add the same environment variables as secrets
4. Set the app file to `app.py`

### Local Production

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🧪 Testing

### Evaluation Data

The system includes a sample evaluation dataset (`evaluation_data.json`) with medical questions and expected keywords. You can:

- Run the evaluation suite from the sidebar
- View detailed results in the Evaluation tab
- Customize evaluation questions for your specific use case

### Custom Evaluation

To add your own evaluation questions:

```json
{
  "question": "Your clinical question here?",
  "expected_keywords": ["keyword1", "keyword2", "keyword3"]
}
```

## 🔍 API Reference

### Core Functions

- `ingest_pdf_to_pinecone(pdf_path)`: Process and index a PDF file
- `retrieve(query, top_k)`: Search for relevant document chunks
- `generate_answer(query, contexts)`: Generate AI response from retrieved context
- `evaluate_retrieval_accuracy(query, expected_keywords)`: Evaluate retrieval performance

### Configuration

- `EMBED_MODEL`: HuggingFace model identifier
- `CHUNK_SIZE`: Maximum chunk size in characters
- `CHUNK_OVERLAP`: Overlap between consecutive chunks
- `TOP_K`: Number of top results to retrieve

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Pinecone](https://www.pinecone.io/) for vector database infrastructure
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for the web interface framework
- [LangChain](https://langchain.com/) for text processing utilities

## 📞 Support

For questions or issues:
- Create an issue in the GitHub repository
- Check the documentation and examples
- Review the configuration options

---

**Note**: This system is designed for educational and research purposes. Always verify medical information with qualified healthcare professionals and authoritative sources.
