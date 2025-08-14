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
_embedder = None
_embedder_cache = {}

def _get_embedder():
    """Get or create the embedding model with caching."""
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
        print("Embedding model loaded successfully!")
    return _embedder

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts with caching for efficiency."""
    embedder = _get_embedder()
    
    # Check cache for individual texts
    results = []
    texts_to_embed = []
    text_indices = []
    
    for i, text in enumerate(texts):
        text_hash = hash(text.lower())
        if text_hash in _embedder_cache:
            results.append(_embedder_cache[text_hash])
        else:
            texts_to_embed.append(text)
            text_indices.append(i)
    
    # Embed texts not in cache
    if texts_to_embed:
        new_embeddings = embedder.encode(texts_to_embed, normalize_embeddings=True).tolist()
        
        # Cache new embeddings and place in results
        for i, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
            text_hash = hash(text.lower())
            _embedder_cache[text_hash] = embedding
            results.insert(text_indices[i], embedding)
    
    return results

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

# ---------- Index Management ----------
def clear_index():
    """Clear all vectors from the Pinecone index."""
    try:
        index = get_index()
        index.delete(delete_all=True)
        print(f"Cleared all vectors from index: {PINECONE_INDEX}")
        return True
    except Exception as e:
        print(f"Failed to clear index: {str(e)}")
        return False

def reindex_pdfs(pdf_directory: str = "data") -> int:
    """Reindex all PDFs in the data directory after clearing the index."""
    try:
        # Clear existing index
        if not clear_index():
            raise Exception("Failed to clear existing index")
        
        # Wait for index to be ready
        time.sleep(5)
        
        # Get all PDF files
        pdf_files = []
        for file in os.listdir(pdf_directory):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(pdf_directory, file))
        
        if not pdf_files:
            print("No PDF files found in data directory")
            return 0
        
        total_chunks = 0
        for pdf_path in pdf_files:
            try:
                print(f"Reindexing: {pdf_path}")
                chunks = ingest_pdf_to_pinecone(pdf_path)
                total_chunks += chunks
                print(f"Indexed {pdf_path}: {chunks} chunks")
            except Exception as e:
                print(f"Failed to index {pdf_path}: {str(e)}")
                continue
        
        print(f"Reindexing complete! Total chunks: {total_chunks}")
        return total_chunks
        
    except Exception as e:
        raise Exception(f"Reindexing failed: {str(e)}")

def get_index_stats() -> Dict:
    """Get statistics about the current index."""
    try:
        index = get_index()
        stats = index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": stats.namespaces
        }
    except Exception as e:
        return {"error": str(e)}

# ---------- Ingestion ----------
def chunk_pdf(pdf_path: str) -> List[Dict]:
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Use more intelligent chunking for medical documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "? ",    # Question endings
                "! ",    # Exclamation endings
                "; ",    # Semicolon separations
                ": ",    # Colon separations
                " ",     # Word boundaries
                ""       # Fallback
            ],
            length_function=len,
            is_separator_regex=False
        )
        chunks = splitter.split_documents(docs)
        
        # Convert to simple dicts with intelligent deduplication
        items = []
        seen_contents = set()  # Track unique content to avoid duplicates
        seen_semantic = set()  # Track semantic similarity
        
        for i, ch in enumerate(chunks):
            content = ch.page_content.strip()
            
            # Skip empty or very short chunks
            if len(content) < 50:
                continue
                
            # Skip chunks that are mostly numbers or special characters
            if len([c for c in content if c.isalnum()]) < len(content) * 0.3:
                continue
                
            # Create content hash for exact deduplication
            content_hash = hash(content.lower())
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)
            
            # Create semantic hash (first and last 50 chars) for similar content
            semantic_key = content[:50] + content[-50:] if len(content) > 100 else content
            semantic_hash = hash(semantic_key.lower())
            if semantic_hash in seen_semantic:
                continue
            seen_semantic.add(semantic_hash)
            
            meta = ch.metadata or {}
            
            # Create more descriptive chunk ID
            chunk_id = f"{meta.get('source', 'pdf')}-{meta.get('page', 0)}-{i:03d}-{content_hash % 10000:04d}"
            
            items.append({
                "id": chunk_id,
                "text": content,
                "metadata": {
                    "source": meta.get("source", pdf_path.split("/")[-1]),
                    "page": meta.get("page", 0),
                    "chunk_index": i,
                    "content_length": len(content),
                    "word_count": len(content.split()),
                    "chunk_type": _classify_chunk_type(content),
                }
            })
        return items
    except Exception as e:
        raise Exception(f"Failed to chunk PDF {pdf_path}: {str(e)}")

def _classify_chunk_type(content: str) -> str:
    """Classify the type of medical content chunk."""
    content_lower = content.lower()
    
    if any(word in content_lower for word in ['table', 'figure', 'chart', 'graph']):
        return 'data_visualization'
    elif any(word in content_lower for word in ['dosage', 'mg', 'ml', 'tablet', 'capsule']):
        return 'dosage_info'
    elif any(word in content_lower for word in ['contraindication', 'warning', 'caution']):
        return 'safety_info'
    elif any(word in content_lower for word in ['mechanism', 'pathophysiology', 'etiology']):
        return 'mechanism'
    elif any(word in content_lower for word in ['symptom', 'sign', 'clinical']):
        return 'clinical_manifestation'
    elif any(word in content_lower for word in ['treatment', 'therapy', 'intervention']):
        return 'treatment'
    else:
        return 'general'

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
        
        # Adaptive retrieval based on query complexity
        query_complexity = _assess_query_complexity(query)
        
        # Adjust top_k based on complexity
        if query_complexity == 'high':
            adaptive_top_k = min(top_k + 5, 20)  # More context for complex queries
        elif query_complexity == 'medium':
            adaptive_top_k = top_k + 2
        else:
            adaptive_top_k = top_k
        
        # First retrieval attempt with higher top_k for diversity
        res = index.query(vector=qvec, top_k=adaptive_top_k * 2, include_metadata=True)
        elapsed = (time.perf_counter() - t0) * 1000.0  # ms

        matches = []
        seen_sources = set()  # Track unique sources to ensure diversity
        source_count = {}  # Track how many chunks from each source
        
        for m in res.matches or []:
            md = m.metadata or {}
            source = md.get("source", "unknown")
            page = md.get("page", 0)
            
            # Create a unique source identifier
            source_key = f"{source}-{page}"
            
            # Limit chunks per source to ensure diversity
            if source_key in seen_sources:
                continue
                
            # Limit total chunks from the same source
            if source_count.get(source, 0) >= 2:
                continue
                
            seen_sources.add(source_key)
            source_count[source] = source_count.get(source, 0) + 1
            
            matches.append({
                "id": m.id,
                "score": float(m.score),
                "text": md.get("text", ""),
                "source": source,
                "page": page,
                "chunk_index": md.get("chunk_index", 0),
                "content_length": md.get("content_length", 0),
            })
            
            # Stop if we have enough diverse results
            if len(matches) >= adaptive_top_k:
                break
        
        # If we don't have enough diverse results, try a different approach
        if len(matches) < top_k:
            try:
                # Try to get more results with different filtering
                res2 = index.query(
                    vector=qvec, 
                    top_k=min(adaptive_top_k * 3, 30), 
                    include_metadata=True,
                    filter={"score": {"$gte": 0.3}}  # Lower threshold for diversity
                )
                
                # Add more diverse results
                for m in res2.matches or []:
                    if len(matches) >= top_k:
                        break
                        
                    md = m.metadata or {}
                    source = md.get("source", "unknown")
                    page = md.get("page", 0)
                    source_key = f"{source}-{page}"
                    
                    # Only add if it's a new source or page
                    if source_key not in seen_sources and source_count.get(source, 0) < 2:
                        seen_sources.add(source_key)
                        source_count[source] = source_count.get(source, 0) + 1
                        
                        matches.append({
                            "id": m.id,
                            "score": float(m.score),
                            "text": md.get("text", ""),
                            "source": source,
                            "page": page,
                            "chunk_index": md.get("chunk_index", 0),
                            "content_length": md.get("content_length", 0),
                        })
            except Exception:
                pass  # Fall back to original results if secondary retrieval fails
        
        # Sort by score and ensure we don't exceed top_k
        matches.sort(key=lambda x: x["score"], reverse=True)
        matches = matches[:top_k]
        
        return matches, elapsed
    except Exception as e:
        raise Exception(f"Retrieval failed: {str(e)}")

# ---------- Generation (Groq) ----------
def _build_prompt(query: str, contexts: List[Dict]) -> str:
    # Analyze query type and context quality to customize the prompt
    query_lower = query.lower()
    context_quality = _assess_context_quality(contexts)
    
    # Sort contexts by relevance and diversity
    sorted_contexts = _organize_contexts_by_relevance(contexts, query_lower)
    
    # Dynamic system rules based on query type
    if any(word in query_lower for word in ['diagnosis', 'diagnose', 'symptoms', 'signs']):
        system_rules = (
            "You are a clinical decision support assistant specializing in diagnostic reasoning. "
            "Use ONLY the provided context to analyze symptoms and suggest possible diagnoses. "
            "Cite sources inline like (source, page). If context is insufficient, say "
            "\"Not enough evidence for diagnosis - consult healthcare provider.\" "
            "Always consider differential diagnoses and note confidence levels."
        )
    elif any(word in query_lower for word in ['treatment', 'therapy', 'medication', 'drug']):
        system_rules = (
            "You are a clinical decision support assistant focusing on treatment options. "
            "Use ONLY the provided context to recommend evidence-based treatments. "
            "Cite sources inline like (source, page). Emphasize contraindications, "
            "drug interactions, and safety considerations. If context lacks treatment info, "
            "say \"Treatment recommendations require more clinical context.\""
        )
    elif any(word in query_lower for word in ['risk', 'complication', 'side effect']):
        system_rules = (
            "You are a clinical decision support assistant assessing risks and complications. "
            "Use ONLY the provided context to identify potential risks and adverse outcomes. "
            "Cite sources inline like (source, page). Prioritize high-risk factors and "
            "prevention strategies. If risk assessment is incomplete, say "
            "\"Insufficient data for comprehensive risk evaluation.\""
        )
    else:
        # Default clinical assistant role
        system_rules = (
            "You are a clinical decision support assistant. Use ONLY the provided context to answer. "
            "Cite sources inline like (source, page). If something is not supported by context, say "
            "\"Not enough evidence in the provided sources.\" Be concise, evidence-based, and note "
            "contraindications or drug interactions if mentioned. Prefer higher quality evidence."
        )
    
    # Adjust prompt based on context quality
    if context_quality == 'high':
        confidence_note = "High-quality evidence available. Provide detailed, confident response."
    elif context_quality == 'medium':
        confidence_note = "Moderate evidence available. Provide balanced response with noted limitations."
    else:
        confidence_note = "Limited evidence available. Provide cautious response with clear limitations."
    
    # Enhanced context formatting with relevance indicators and chunk types
    contexts_str = "\n\n".join([
        f"[Source: {c['source']}, page {c.get('page')} | Type: {c.get('chunk_type', 'general')} | Relevance: {c['score']:.3f}]\n{c['text']}" 
        for c in sorted_contexts
    ])
    
    # Add query-specific instructions
    query_instructions = _get_query_specific_instructions(query_lower)
    
    return (
        f"{system_rules}\n\n"
        f"Context Quality: {confidence_note}\n\n"
        f"Context:\n{contexts_str}\n\n"
        f"Query: {query}\n\n"
        f"{query_instructions}\n\n"
        f"Answer:"
    )

def _organize_contexts_by_relevance(contexts: List[Dict], query: str) -> List[Dict]:
    """Organize contexts by relevance and diversity for better prompt construction."""
    if not contexts:
        return []
    
    # Score contexts based on query relevance and diversity
    scored_contexts = []
    for ctx in contexts:
        score = ctx['score']
        
        # Bonus for diverse source types
        if ctx.get('chunk_type') != 'general':
            score += 0.1
            
        # Bonus for different sources
        source_bonus = 0.05 if len(set(c['source'] for c in contexts)) > 1 else 0
        
        scored_contexts.append((ctx, score + source_bonus))
    
    # Sort by enhanced score
    scored_contexts.sort(key=lambda x: x[1], reverse=True)
    
    # Return top contexts with diversity
    result = []
    seen_types = set()
    seen_sources = set()
    
    for ctx, _ in scored_contexts:
        if len(result) >= 5:  # Limit to top 5 most relevant
            break
            
        # Ensure diversity in chunk types and sources
        chunk_type = ctx.get('chunk_type', 'general')
        source = ctx.get('source', 'unknown')
        
        if chunk_type not in seen_types or source not in seen_sources:
            result.append(ctx)
            seen_types.add(chunk_type)
            seen_sources.add(source)
        elif len(result) < 3:  # Always include top 3
            result.append(ctx)
    
    return result

def _assess_context_quality(contexts: List[Dict]) -> str:
    """Assess the quality of retrieved contexts based on relevance scores."""
    if not contexts:
        return 'low'
    
    avg_score = sum(c['score'] for c in contexts) / len(contexts)
    
    if avg_score >= 0.8:
        return 'high'
    elif avg_score >= 0.6:
        return 'medium'
    else:
        return 'low'

def _get_query_specific_instructions(query: str) -> str:
    """Provide query-specific instructions to guide the response."""
    if any(word in query for word in ['compare', 'difference', 'versus', 'vs']):
        return "Provide a structured comparison highlighting key differences and similarities."
    elif any(word in query for word in ['how', 'procedure', 'technique']):
        return "Provide step-by-step guidance if available in the context."
    elif any(word in query for word in ['when', 'timing', 'schedule']):
        return "Focus on temporal aspects and timing considerations mentioned in the context."
    elif any(word in query for word in ['why', 'cause', 'reason']):
        return "Explain the underlying mechanisms and reasoning based on available evidence."
    else:
        return "Provide a comprehensive, evidence-based response using the available context."

def generate_answer(query: str, contexts: List[Dict]) -> Tuple[str, float]:
    try:
        t0 = time.perf_counter()
        client = Groq(api_key=GROQ_API_KEY)
        prompt = _build_prompt(query, contexts)
        
        # Dynamic model selection and parameters based on query complexity
        query_complexity = _assess_query_complexity(query)
        context_quality = _assess_context_quality(contexts)
        
        # Choose model based on complexity
        if query_complexity == 'high' or context_quality == 'low':
            model = "llama3-70b-8192"  # More capable model for complex queries
            max_tokens = 800
            temperature = 0.2  # Slightly more creative for complex reasoning
        else:
            model = "llama3-8b-8192"  # Faster model for simpler queries
            max_tokens = 600
            temperature = 0.1  # More deterministic for straightforward queries
        
        # Adjust parameters based on context quality
        if context_quality == 'low':
            max_tokens = min(max_tokens + 200, 1000)  # Allow longer responses for limited context
            temperature = min(temperature + 0.1, 0.3)  # Slightly more flexible
        
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = resp.choices[0].message.content
        elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        return answer, elapsed
    except Exception as e:
        raise Exception(f"Generation failed: {str(e)}")

def _assess_query_complexity(query: str) -> str:
    """Assess the complexity of a query based on various factors."""
    query_lower = query.lower()
    
    # Count complexity indicators
    complexity_score = 0
    
    # Medical terminology complexity
    medical_terms = ['differential diagnosis', 'pathophysiology', 'mechanism', 'etiology', 
                     'pathogenesis', 'molecular', 'genetic', 'biochemical']
    complexity_score += sum(2 for term in medical_terms if term in query_lower)
    
    # Query structure complexity
    if any(word in query_lower for word in ['compare', 'analyze', 'evaluate', 'assess']):
        complexity_score += 2
    if any(word in query_lower for word in ['why', 'how', 'mechanism']):
        complexity_score += 1
    if '?' in query:
        complexity_score += 1
    
    # Length and structure
    if len(query.split()) > 15:
        complexity_score += 1
    if any(char in query for char in ['(', ')', ';', ':']):
        complexity_score += 1
    
    if complexity_score >= 4:
        return 'high'
    elif complexity_score >= 2:
        return 'medium'
    else:
        return 'low'

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
