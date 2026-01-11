from collections import defaultdict
from typing import List, Tuple


def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if min_s == max_s:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridRetriever:
    def __init__(self, bm25_retriever, vectorstore, alpha: float = 0.6):
        """
        alpha: weight for semantic search (0..1)
        (1 - alpha): weight for BM25
        """
        self.bm25 = bm25_retriever
        self.vs = vectorstore
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5):
        # --- BM25 ---
        bm25_docs = self.bm25.search(query, top_k=top_k * 2)
        bm25_scores = list(range(len(bm25_docs), 0, -1))  # rank-based scores
        bm25_scores = min_max_normalize(bm25_scores)

        # --- Semantic ---
        sem_results = self.vs.similarity_search_with_score(query, k=top_k * 2)
        sem_docs = [doc for doc, _ in sem_results]
        sem_raw_scores = [score for _, score in sem_results]
        # Chroma returns distance; convert to similarity
        sem_scores = min_max_normalize([1 / (1 + s) for s in sem_raw_scores])

        # --- Fuse ---
        fused = defaultdict(float)
        doc_map = {}

        for doc, score in zip(bm25_docs, bm25_scores):
            key = doc.page_content
            fused[key] += (1 - self.alpha) * score
            doc_map[key] = doc

        for doc, score in zip(sem_docs, sem_scores):
            key = doc.page_content
            fused[key] += self.alpha * score
            doc_map[key] = doc

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, _ in ranked[:top_k]]
