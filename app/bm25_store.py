from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        """
        documents: list of LangChain Document objects
        """
        self.documents = documents
        self.tokenized_docs = [
            doc.page_content.lower().split() for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5):
        """
        Return top_k documents based on BM25 scores
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]
