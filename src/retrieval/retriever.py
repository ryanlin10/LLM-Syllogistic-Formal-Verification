"""Retrieval system for grounding premises with evidence."""

import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    index_path: str = "./data/retrieval_index"
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50


class DocumentChunk:
    """A chunk of a document with metadata."""
    def __init__(self, doc_id: str, text: str, start: int, end: int, metadata: Optional[Dict] = None):
        self.doc_id = doc_id
        self.text = text
        self.start = start
        self.end = end
        self.metadata = metadata or {}


class DocumentRetriever:
    """Dense retriever using sentence transformers and FAISS."""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
        self.index = None
        self.chunks = []  # Store original chunks
        self.doc_metadata = {}  # Store document metadata
        
        Path(config.index_path).mkdir(parents=True, exist_ok=True)
    
    def chunk_document(self, doc_id: str, text: str) -> List[DocumentChunk]:
        """Split document into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                text=chunk_text,
                start=start,
                end=end
            ))
            
            start += self.config.chunk_size - self.config.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def index_documents(self, documents: Dict[str, str], metadata: Optional[Dict[str, Dict]] = None):
        """Index a collection of documents."""
        print("Chunking documents...")
        all_chunks = []
        
        for doc_id, text in documents.items():
            chunks = self.chunk_document(doc_id, text)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        self.doc_metadata = metadata or {}
        
        print(f"Chunked into {len(all_chunks)} chunks")
        print("Computing embeddings...")
        
        # Compute embeddings
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        print(f"Indexed {len(all_chunks)} chunks in FAISS index")
        
        # Save index
        self.save_index()
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-k relevant document chunks."""
        if self.index is None:
            raise ValueError("Index not built. Call index_documents() first.")
        
        top_k = top_k or self.config.top_k
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append((chunk, float(dist)))
        
        return results
    
    def retrieve_for_context(self, question: str, top_k: Optional[int] = None) -> str:
        """Retrieve and format context for model input."""
        results = self.retrieve(question, top_k)
        
        context_parts = []
        for chunk, score in results:
            context_parts.append(f"[Doc: {chunk.doc_id}, Score: {score:.3f}]\n{chunk.text}")
        
        return "\n\n".join(context_parts)
    
    def link_premise_to_evidence(
        self,
        premise: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Link a premise to evidence spans."""
        results = self.retrieve(premise, top_k)
        
        evidence_spans = []
        for chunk, score in results:
            if score > 0.5:  # Threshold for relevance
                evidence_spans.append({
                    "doc_id": chunk.doc_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "score": score
                })
        
        return evidence_spans
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        index_file = Path(self.config.index_path) / "index.faiss"
        chunks_file = Path(self.config.index_path) / "chunks.pkl"
        metadata_file = Path(self.config.index_path) / "metadata.pkl"
        
        if self.index is not None:
            faiss.write_index(self.index, str(index_file))
            print(f"Saved FAISS index to {index_file}")
        
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(metadata_file, "wb") as f:
            pickle.dump(self.doc_metadata, f)
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        index_file = Path(self.config.index_path) / "index.faiss"
        chunks_file = Path(self.config.index_path) / "chunks.pkl"
        metadata_file = Path(self.config.index_path) / "metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found at {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        print(f"Loaded FAISS index from {index_file}")
        
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        
        with open(metadata_file, "rb") as f:
            self.doc_metadata = pickle.load(f)
        
        print(f"Loaded {len(self.chunks)} chunks and metadata for {len(self.doc_metadata)} documents")

