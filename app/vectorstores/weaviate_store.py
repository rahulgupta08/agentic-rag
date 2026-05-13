import asyncio

from app.vectorstores.base_store import BaseVectorStore
from app.ingestion.ingestion_utils import generate_deterministic_uuid


class WeaviateStore(BaseVectorStore):

    def __init__(self, client, collection_name: str, namespace: str = None):
        self.collection = client.collections.get(collection_name)
        self.namespace = namespace

    # ---------------- INGESTION ----------------

    def upsert(self, vectors, namespace=None):
        """Synchronous upsert for RAG mode"""
        with self.collection.batch.dynamic() as batch:
            for item in vectors:
                properties = {
                    "text": item["text"],
                    **item.get("metadata", {})
                }
                uuid_value = generate_deterministic_uuid(item["id"])
                batch.add_object(
                    uuid=uuid_value,
                    properties=properties,
                    vector=item["vector"]
                )

    async def aupsert(self, vectors, namespace=None):
        """Asynchronous upsert for Agent mode"""
        await asyncio.to_thread(self.upsert, vectors, namespace)

    def delete(self, ids):
        """Synchronous delete for RAG mode"""
        for id_ in ids:
            self.collection.data.delete_by_id(id_)

    async def adelete(self, ids):
        """Asynchronous delete for Agent mode"""
        await asyncio.to_thread(self.delete, ids)

    def update(self, id, vector=None, metadata=None):
        """Synchronous update for RAG mode"""
        if metadata:
            self.collection.data.update(
                uuid=id,
                properties=metadata
            )

        if vector:
            self.collection.data.update(
                uuid=id,
                vector=vector
            )

    async def aupdate(self, id, vector=None, metadata=None):
        """Asynchronous update for Agent mode"""
        await asyncio.to_thread(self.update, id, vector, metadata)

    # ---------------- RETRIEVAL ----------------

    def dense_search(self, query_vector, top_k=5, namespace=None):
        """Synchronous dense search for RAG mode"""
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=int(top_k),
            return_metadata=["distance"]
        )
        return self._process_search_results(response)

    async def dense_search_async(self, query_vector, top_k=5, namespace=None):
        """Asynchronous dense search for Agent mode"""
        response = await asyncio.to_thread(
            self.collection.query.near_vector,
            near_vector=query_vector,
            limit=int(top_k),
            return_metadata=["distance"]
        )
        return self._process_search_results(response)

    def hybrid_search(self, query_text, top_k=5, alpha=0.5):
        """Synchronous hybrid search for RAG mode"""
        response = self.collection.query.hybrid(
            query=query_text,
            alpha=alpha,
            limit=int(top_k),
            return_metadata=["distance"]
        )
        return self._process_search_results(response)

    async def hybrid_search_async(self, query_text, top_k=5, alpha=0.5):
        """Asynchronous hybrid search for Agent mode"""
        response = await asyncio.to_thread(
            self.collection.query.hybrid,
            query=query_text,
            alpha=alpha,
            limit=int(top_k),
            return_metadata=["distance"]
        )
        return self._process_search_results(response)

    def search(self, query_vector, top_k):
        """Synchronous search for RAG mode"""
        return self.dense_search(query_vector, top_k)

    async def search_async(self, query_vector, top_k):
        """Asynchronous search for Agent mode"""
        return await self.dense_search_async(query_vector, top_k)

    def _process_search_results(self, response):
        """Helper method to process search results into standard format"""
        results = []
        for obj in response.objects:
            distance = getattr(obj.metadata, "distance", None)
            score = 1 - distance if distance is not None else None

            results.append({
                "id": str(obj.uuid),
                "text": obj.properties.get("text", ""),
                "metadata": obj.properties,
                "score": score
            })
        return results