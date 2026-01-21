import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class HybridRAG:
    def __init__(self, retriever):
        self.retriever = retriever

        # Read Hugging Face token from environment (Streamlit Secrets)
        hf_token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-small",
            token=hf_token
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small",
            token=hf_token
        )

    def answer(self, query: str) -> str:
        docs = self.retriever.search(query, top_k=5)
        context = "\n".join(docs)

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
