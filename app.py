import os
import time
import json
import streamlit as st
from rag import (
    ingest_pdf_to_pinecone, retrieve, generate_answer, simple_retrieval_score,
    evaluate_retrieval_accuracy, run_evaluation_suite
)
from config import SHOW_RETRIEVED_CHUNKS

from dotenv import load_dotenv
load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

st.set_page_config(page_title="Medical Literature RAG", page_icon="🩺", layout="wide")
st.title("🩺 Medical Literature RAG (Evidence-Based)")

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Sidebar for file upload and evaluation
with st.sidebar:
    st.header("📁 Upload medical PDFs")
    os.makedirs("data", exist_ok=True)
    up = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True)
    if up:
        total_chunks = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("Indexing..."):
                for i, f in enumerate(up):
                    status_text.text(f"Processing {f.name}...")
                    progress_bar.progress((i) / len(up))
                    
                    path = os.path.join("data", f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    
                    # Index the PDF with progress updates
                    try:
                        chunks = ingest_pdf_to_pinecone(path)
                        total_chunks += chunks
                        status_text.text(f"Indexed {f.name}: {chunks} chunks")
                        progress_bar.progress((i + 1) / len(up))
                    except Exception as e:
                        st.error(f"Failed to index {f.name}: {str(e)}")
                        continue
                
                progress_bar.progress(1.0)
                status_text.text("Indexing complete!")
                st.success(f"Ingested {len(up)} file(s), {total_chunks} chunks.")
                
        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")
            st.info("Please check your API keys and internet connection.")
    
    st.markdown("---")
    st.header("🧪 Evaluation")
    
    # Load evaluation data
    try:
        with open("evaluation_data.json", "r") as f:
            evaluation_data = json.load(f)
        st.write(f"Loaded {len(evaluation_data)} evaluation questions")
        
        if st.button("Run Evaluation Suite"):
            with st.spinner("Running evaluation..."):
                try:
                    results = run_evaluation_suite(evaluation_data)
                    st.session_state.evaluation_results = results
                    st.success("Evaluation complete!")
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    
    except FileNotFoundError:
        st.warning("evaluation_data.json not found")
        evaluation_data = []

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Query & Generate", "📊 Evaluation Results", "📚 About"])

with tab1:
    st.subheader("Ask a clinical question")
    query = st.text_input("Example: First-line therapy for stage 2 hypertension in a 60-year-old with diabetes?")
    go = st.button("Search & Generate")

    if go and query.strip():
        try:
            with st.spinner("Retrieving…"):
                matches, rt_ms = retrieve(query, top_k=5)
            if not matches:
                st.warning("No relevant passages found. Try another query or upload more literature.")
            else:
                avg_score = simple_retrieval_score(matches)

                cols = st.columns([2, 1])
                with cols[0]:
                    with st.spinner("Generating answer…"):
                        try:
                            answer, gen_ms = generate_answer(query, matches)
                            st.markdown("### 💡 Evidence-based answer")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Failed to generate answer: {str(e)}")
                            answer = "Error generating answer. Please check your Groq API key."
                            gen_ms = 0

                with cols[1]:
                    st.markdown("### 📊 Diagnostics")
                    st.write(f"Retrieval latency: **{rt_ms:.0f} ms**")
                    st.write(f"Generation latency: **{gen_ms:.0f} ms**")
                    st.write(f"Avg retrieval score: **{avg_score:.3f}** (cosine)")

                st.markdown("---")
                st.markdown("### 📚 Sources")
                for i, m in enumerate(matches, 1):
                    st.markdown(
                        f"**{i}. {m['source']}** (page {m.get('page')}) — score {m['score']:.3f}\n\n"
                        + (f"> {m['text'][:700]}..." if SHOW_RETRIEVED_CHUNKS else "")
                    )
        except Exception as e:
            st.error(f"Query failed: {str(e)}")
            st.info("Please check your Pinecone connection and API keys.")

with tab2:
    st.subheader("Evaluation Results")
    
    if st.session_state.evaluation_results:
        results = st.session_state.evaluation_results
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{results['overall_accuracy']:.1%}")
        with col2:
            st.metric("Avg Latency", f"{results['overall_latency_ms']:.0f} ms")
        with col3:
            st.metric("Total Questions", results['total_questions'])
        
        # Detailed results
        st.markdown("### Detailed Results")
        for i, item in enumerate(results['detailed_results']):
            with st.expander(f"Q{i+1}: {item['question']}"):
                result = item['result']
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.write(f"**Accuracy:** {result['accuracy']:.1%}")
                    st.write(f"**Latency:** {result['latency_ms']:.0f} ms")
                    st.write(f"**Keywords Found:** {result['keywords_found']}/{result['total_keywords']}")
                    
                    if result['found_keywords']:
                        st.write(f"**Found:** {', '.join(result['found_keywords'])}")
                    if result['missing_keywords']:
                        st.write(f"**Missing:** {', '.join(result['missing_keywords'])}")
    else:
        st.info("Run the evaluation suite from the sidebar to see results here.")

with tab3:
    st.subheader("About This RAG System")
    st.markdown("""
    This Medical Literature RAG system provides evidence-based answers to clinical questions using:
    
    - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
    - **Vector Database**: Pinecone for similarity search
    - **Chunking**: Recursive text splitting optimized for medical documents
    - **Generation**: Groq LLM (Llama3-8B) with context-aware prompting
    - **Evaluation**: Retrieval accuracy metrics and latency tracking
    
    ### Features:
    - PDF ingestion and chunking
    - Semantic search with relevance scoring
    - Evidence-based answer generation
    - Source citation and page references
    - Performance diagnostics and evaluation
    
    ### Use Cases:
    - Clinical decision support
    - Medical literature review
    - Evidence-based medicine research
    - Medical education and training
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, Pinecone, and Groq*")
