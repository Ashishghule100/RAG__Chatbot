import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.rag.vector_store import VectorStore
from src.rag.llm import LLM

class RAGSystem:
    def __init__(self, model_path: str = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = VectorStore()
        self.llm = LLM(model_path=model_path)

    def _get_embedding(self, text: str) -> List[float]:
        return self.embed_model.encode(text).tolist()

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, str]] = None) -> List[Dict]:
        q_emb = self._get_embedding(query)
        return self.vector_store.search(q_emb, top_k=top_k, filters=filters)

    def build_prompt(self, contexts: List[Dict], question: str) -> str:
        context_texts = []
        for ctx in contexts:
            text = ctx.get("text_content", "")[:1000]
            meta = ctx.get("metadata", {})
            book = meta.get("bookname", "Unknown")
            page = meta.get("page_no", "?")
            context_texts.append(f"[{book}, page {page}]\n{text}")
        combined = "\n\n---\n\n".join(context_texts)
        prompt = f"""You are a helpful teaching assistant. Answer the question based only on the provided context.
If the answer is not in the context, say "I don't know". Cite sources by book name and page number.

Context:
{combined}

Question: {question}

Answer:"""
        return prompt

    def answer(self, question: str, filters: Optional[Dict[str, str]] = None, top_k: int = 8) -> Dict[str, Any]:
        results = self.search(question, top_k=top_k, filters=filters)
        if not results:
            return {"answer": "No relevant information found.", "sources": []}

        prompt = self.build_prompt(results, question)
        answer = self.llm.generate(prompt, max_tokens=512)

        sources = []
        for r in results[:3]:
            meta = r.get("metadata", {})
            sources.append({
                "bookname": meta.get("bookname", "Unknown"),
                "page_no": meta.get("page_no", "?")
            })
        return {"answer": answer, "sources": sources}

    def close(self):
        self.vector_store.close()