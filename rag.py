import os
import time
import uuid
from typing import List, Tuple, Dict

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone, ServerlessSpec
from groq import Groq

from config import (
    PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL, EMBED_DIM,
    TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, GROQ_API_KEY
)

# ---------- Embedding model ----------
_embedder = SentenceTransformer(EMBED_MODEL)

def embed_texts(texts: List[str]) -> List[List[float]]:
    return _embedder.encode(texts, normalize_embeddings=True).tolist()

# ---------- Pinecone setup ----------
def _ensure_pinecone_index():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = {idx["name"] for idx in pc.list_indexes()}
        
        if PINECONE_INDEX not in existing:
            print(f"Creating new index: {PINECONE_INDEX} with {EMBED_DIM} dimensions")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait a moment for index to be ready
            time.sleep(5)
        else:
            print(f"Using existing index: {PINECONE_INDEX}")
            
        return pc.Index(PINECONE_INDEX)
    except Exception as e:
        raise Exception(f"Failed to connect to Pinecone: {str(e)}")

_index = None
def get_index():
    global _index
    if _index is None:
        _index = _ensure_pinecone_index()
    return _index

# ---------- Ingestion ----------
def chunk_pdf(pdf_path: str) -> List[Dict]:
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        # Convert to simple dicts
        items = []
        for i, ch in enumerate(chunks):
            meta = ch.metadata or {}
            items.append({
                "id": f"{meta.get('source', 'pdf')}-{meta.get('page', 0)}-{i}-{uuid.uuid4().hex[:8]}",
                "text": ch.page_content.strip(),
                "metadata": {
                    "source": meta.get("source", pdf_path.split("/")[-1]),
                    "page": meta.get("page", 0),
                }
            })
        return items
    except Exception as e:
        raise Exception(f"Failed to chunk PDF {pdf_path}: {str(e)}")

def ingest_pdf_to_pinecone(pdf_path: str) -> int:
    try:
        # Step 1: Chunk the PDF
        items = chunk_pdf(pdf_path)
        if not items:
            raise Exception("No chunks created from PDF")
        
        # Step 2: Create embeddings
        texts = [it["text"] for it in items]
        vecs = embed_texts(texts)
        
        # Step 3: Get Pinecone index
        index = get_index()
        
        # Step 4: Upsert in smaller batches with timeout protection
        batch_size = 50  # Reduced from 100
        total_upserted = 0
        
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_vecs = vecs[i:i + batch_size]
            
            batch = []
            for it, vec in zip(batch_items, batch_vecs):
                md = {"text": it["text"], **it["metadata"]}
                batch.append({"id": it["id"], "values": vec, "metadata": md})
            
            # Upsert with timeout
            try:
                index.upsert(vectors=batch)
                total_upserted += len(batch)
                time.sleep(0.1)  # Small delay between batches
            except Exception as e:
                raise Exception(f"Failed to upsert batch {i//batch_size + 1}: {str(e)}")
        
        return total_upserted
        
    except Exception as e:
        raise Exception(f"Indexing failed: {str(e)}")

# ---------- Retrieval ----------
def retrieve(query: str, top_k: int = TOP_K) -> Tuple[List[Dict], float]:
    try:
        t0 = time.perf_counter()
        qvec = embed_texts([query])[0]
        index = get_index()
        res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
        elapsed = (time.perf_counter() - t0) * 1000.0  # ms

        matches = []
        for m in res.matches or []:
            md = m.metadata or {}
            matches.append({
                "id": m.id,
                "score": float(m.score),
                "text": md.get("text", ""),
                "source": md.get("source", "unknown"),
                "page": md.get("page", None),
            })
        return matches, elapsed
    except Exception as e:
        raise Exception(f"Retrieval failed: {str(e)}")

# ---------- Generation (Groq) ----------
def _build_prompt(query: str, contexts: List[Dict]) -> str:
    contexts_str = "\n\n".join(
        [f"[Source: {c['source']}, page {c.get('page')} | score {c['score']:.3f}]\n{c['text']}" for c in contexts]
    )
    system_rules = (
        "You are a clinical decision support assistant. Use ONLY the provided context to answer. "
        "Cite sources inline like (source, page). If something is not supported by context, say "
        "\"Not enough evidence in the provided sources.\" Be concise, evidence-based, and note "
        "contraindications or drug interactions if mentioned. Prefer higher quality evidence."
    )
    return (
        f"{system_rules}\n\n"
        f"Context:\n{contexts_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

def generate_answer(query: str, contexts: List[Dict]) -> Tuple[str, float]:
    try:
        t0 = time.perf_counter()
        client = Groq(api_key=GROQ_API_KEY)
        prompt = _build_prompt(query, contexts)
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )
        answer = resp.choices[0].message.content
        elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        return answer, elapsed
    except Exception as e:
        raise Exception(f"Generation failed: {str(e)}")

# ---------- Simple evaluation helper ----------
def simple_retrieval_score(matches: List[Dict]) -> float:
    """Average similarity score of returned chunks (0..1). Higher ~ better."""
    if not matches:
        return 0.0
    return sum(m["score"] for m in matches) / len(matches)

# ---------- Evaluation functions ----------
def evaluate_retrieval_accuracy(query: str, expected_keywords: List[str], top_k: int = 5) -> Dict:
    """
    Evaluate retrieval accuracy by checking if expected keywords appear in retrieved chunks.
    
    Args:
        query: The search query
        expected_keywords: List of keywords that should appear in relevant chunks
        top_k: Number of top chunks to retrieve
        
    Returns:
        Dict with accuracy metrics
    """
    try:
        matches, latency = retrieve(query, top_k=top_k)
        
        if not matches:
            return {
                "accuracy": 0.0,
                "latency_ms": latency,
                "chunks_retrieved": 0,
                "keywords_found": 0,
                "total_keywords": len(expected_keywords)
            }
        
        # Check how many expected keywords appear in retrieved chunks
        found_keywords = set()
        for match in matches:
            chunk_text = match['text'].lower()
            for keyword in expected_keywords:
                if keyword.lower() in chunk_text:
                    found_keywords.add(keyword.lower())
        
        accuracy = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
        
        return {
            "accuracy": accuracy,
            "latency_ms": latency,
            "chunks_retrieved": len(matches),
            "keywords_found": len(found_keywords),
            "total_keywords": len(expected_keywords),
            "found_keywords": list(found_keywords),
            "missing_keywords": [kw for kw in expected_keywords if kw.lower() not in found_keywords]
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "latency_ms": 0.0,
            "chunks_retrieved": 0,
            "keywords_found": 0,
            "total_keywords": len(expected_keywords),
            "error": str(e)
        }

def run_evaluation_suite(evaluation_data: List[Dict]) -> Dict:
    """
    Run a complete evaluation suite on multiple Q&A pairs.
    
    Args:
        evaluation_data: List of dicts with 'question' and 'expected_keywords' keys
        
    Returns:
        Dict with overall evaluation metrics
    """
    results = []
    total_accuracy = 0.0
    total_latency = 0.0
    
    for item in evaluation_data:
        result = evaluate_retrieval_accuracy(
            item['question'], 
            item['expected_keywords']
        )
        results.append({
            'question': item['question'],
            'result': result
        })
        total_accuracy += result['accuracy']
        total_latency += result['latency_ms']
    
    num_questions = len(evaluation_data)
    avg_accuracy = total_accuracy / num_questions if num_questions > 0 else 0.0
    avg_latency = total_latency / num_questions if num_questions > 0 else 0.0
    
    return {
        'overall_accuracy': avg_accuracy,
        'overall_latency_ms': avg_latency,
        'total_questions': num_questions,
        'detailed_results': results
    }
