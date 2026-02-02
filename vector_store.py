import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index, embeddings, text_chunks


def search_index(index, query, text_chunks, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = [text_chunks[i] for i in indices[0]]
    return results
