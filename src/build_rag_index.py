"""
Construit l'index RAG ChromaDB a partir des snippets medicaux.
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
SNIPPETS_PATH = PROJECT_DIR / "data" / "guidelines" / "snippets_ic_fr.jsonl"
CHROMA_DIR = PROJECT_DIR / "data" / "rag_index"
COLLECTION_NAME = "guidelines_ic"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_snippets():
    snippets = []
    with SNIPPETS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snippets.append(json.loads(line))
    return snippets


def main():
    snippets = load_snippets()
    if not snippets:
        raise RuntimeError("Aucun snippet trouve.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    documents = [s["text"] for s in snippets]
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[s["id"] for s in snippets],
        documents=documents,
        embeddings=embeddings,
        metadatas=[{"source": s["source"], "topic": s["topic"]} for s in snippets],
    )

    print(f"Index RAG construit: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Snippets indexes: {len(snippets)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
