import os
import json
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "textbook_chunks"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ VectorStore ready (collection: {self.collection_name})")

    def upsert_chunk(self, chunk_data: Dict[str, Any]) -> bool:
        """Add or update a single chunk. Returns True on success."""
        try:
            chunk_id = chunk_data["chunk_id"]
            embedding = chunk_data["embedding"]
            # Chroma expects list of floats
            metadata = {
                k: v for k, v in chunk_data.items()
                if k not in ("chunk_id", "embedding", "text_content")
            }
            # Store text_content separately; Chroma can store it in metadata
            metadata["text_content"] = chunk_data.get("text_content", "")
            # Ensure all values are strings, ints, floats, or bools
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    metadata[k] = json.dumps(v)
            self.collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"❌ VectorStore upsert error: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 10,
               filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Search by embedding with optional metadata filters."""
        try:
            where = None
            if filters:
                # Chroma supports filtering by metadata
                where = {k: v for k, v in filters.items() if v is not None}
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["metadatas", "distances"]
            )
            chunks = []
            if results["ids"] and results["ids"][0]:
                for idx, cid in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][idx]
                    # Deserialize any JSON fields
                    for k, v in meta.items():
                        if isinstance(v, str) and v.startswith(("{", "[")):
                            try:
                                meta[k] = json.loads(v)
                            except:
                                pass
                    chunks.append({
                        "chunk_id": cid,
                        "text_content": meta.pop("text_content", ""),
                        "metadata": meta,
                        "distance": results["distances"][0][idx]
                    })
            return chunks
        except Exception as e:
            print(f"❌ VectorStore search error: {e}")
            return []

    def get_all_chunks(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve chunks for evaluation or inspection."""
        try:
            results = self.collection.get(limit=limit, include=["metadatas"])
            chunks = []
            if results["ids"]:
                for idx, cid in enumerate(results["ids"]):
                    meta = results["metadatas"][idx]
                    # Deserialize if needed
                    for k, v in meta.items():
                        if isinstance(v, str) and v.startswith(("{", "[")):
                            try:
                                meta[k] = json.loads(v)
                            except:
                                pass
                    chunks.append({
                        "chunk_id": cid,
                        "text_content": meta.pop("text_content", ""),
                        "metadata": meta
                    })
            return chunks
        except Exception as e:
            print(f"❌ VectorStore get_all error: {e}")
            return []

    def delete_book(self, bookname: str) -> bool:
        """Delete all chunks belonging to a book."""
        try:
            self.collection.delete(where={"bookname": bookname})
            return True
        except Exception as e:
            print(f"❌ Delete error: {e}")
            return False

    def close(self):
        # Chroma client doesn't require explicit close
        pass