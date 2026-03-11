from app.vectorstores.base_store import BaseVectorStore


class PineconeStore(BaseVectorStore):

    def __init__(self, index, namespace: str = None):
        self.index = index
        self.namespace = namespace


    # ---------------- INGESTION ----------------

    def upsert(self, vectors, namespace=None):

        formatted = []

        for item in vectors:
            formatted.append((
                item["id"],
                item["vector"],
                {
                    "text": item["text"],
                    **item.get("metadata", {})
                }
            ))

        self.index.upsert(vectors=formatted, namespace=namespace )


    def delete(self, ids):
        self.index.delete(ids=ids)


    def update(self, id, vector=None, metadata=None):

        # Pinecone does not have native update —
        # must re-upsert
        existing = self.index.fetch(ids=[id])

        if id not in existing["vectors"]:
            return

        old = existing["vectors"][id]

        new_vector = vector if vector else old["values"]
        new_metadata = metadata if metadata else old["metadata"]

        self.index.upsert(
            vectors=[(id, new_vector, new_metadata)]
        )


    # ---------------- RETRIEVAL ----------------

    def dense_search(self, query_vector, top_k=5, namespace=None):

        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

        results = []

        for match in response["matches"]:
            results.append({
                "id": match["id"],
                "text": match["metadata"].get("text", ""),
                "metadata": match["metadata"],
                "score": match["score"]
            })

        return results


    def hybrid_search(self, query_text, top_k=5, alpha=0.5):
        raise NotImplementedError(
            "Hybrid search not supported in PineconeStore yet."
        )
    
    # This is foe querying the vector store directly without LLM generation. Used in test_simple_rag.py
    def search(self, query_vector, top_k):

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        documents = []

        for match in results.matches:
            documents.append({
                "id": match.id,
                "text": match.metadata["text"],
                "score": match.score
            })

        return documents