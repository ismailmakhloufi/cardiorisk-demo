"""
Teste la recherche RAG sur une requete clinique.
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "data" / "rag_index"
COLLECTION_NAME = "guidelines_ic"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def search(query, n_results=5):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)


def main():
    query = (
        "Patient insuffisant cardiaque avec risque de rehospitalisation, "
        "FEVG basse, creatinine elevee, antecedent d'hospitalisation, "
        "besoin de recommandations de sortie et suivi ambulatoire."
    )
    results = search(query, n_results=6)
    print("Requete:")
    print(query)
    print("\nResultats RAG:")
    for i, doc in enumerate(results["documents"][0], start=1):
        meta = results["metadatas"][0][i - 1]
        dist = results["distances"][0][i - 1]
        print(f"\n{i}. topic={meta['topic']} | source={meta['source']} | distance={dist:.4f}")
        print(doc)


if __name__ == "__main__":
    main()
