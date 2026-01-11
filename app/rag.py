from transformers import pipeline


class HybridRAG:
    def __init__(self, hybrid_retriever):
        self.retriever = hybrid_retriever
        self.llm = pipeline(
            "text-generation",
            model="google/flan-t5-small",
            max_new_tokens=200
        )

    def answer(self, query: str, top_k: int = 3) -> str:
        docs = self.retriever.search(query, top_k=top_k)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        response = self.llm(prompt)[0]["generated_text"]
        return response.strip()
