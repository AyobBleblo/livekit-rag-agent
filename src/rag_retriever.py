"""
rag_retriever.py — Retrieves the closest chunks from ChromaDB based on the user's query.
Used by agent.py during the voice conversation.
"""

import os
from pathlib import Path
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path("chroma_db")

# Lazy-load everything so importing this module is instant.
_genai_client = None
_chroma_client = None
_collection = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("[ERROR] GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                "[ERROR] chroma_db/ folder not found. Run build_rag.py first to build the database."
            )
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_collection(name="clinic_knowledge")
    return _collection


def retrieve(query: str, n_results: int = 3) -> str:
    """Embed the query and return the top-n matching chunks as a single string."""
    response = _get_genai_client().models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    query_embedding = response.embeddings[0].values

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    chunks = results["documents"][0]
    return "\n\n---\n\n".join(chunks)