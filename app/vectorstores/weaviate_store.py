import asyncio

from app.vectorstores.base_store import BaseVectorStore
from app.ingestion.ingestion_utils import generate_deterministic_uuid


class WeaviateStore(BaseVectorStore):

    def __init__(self, client, collection_name: str, namespace: str = None):
        self.collection = client.collections.get(collection_name)
        self.namespace = namespace


    # ---------------- INGESTION ----------------

    def upsert(self, vectors, namespace=None):
        """
        vectors format:
        [
            {
                "id": "uuid",
                "vector": [...],
                "text": "...",
                "metadata": {...}
            }
        ]
        """
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


    def delete(self, ids):
        for id_ in ids:
            self.collection.data.delete_by_id(id_)


    def update(self, id, vector=None, metadata=None):

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


    # ---------------- RETRIEVAL ----------------

    async def dense_search(self, query_vector, top_k=5, namespace=None):
        
        def _search():
            return self.collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=["distance"]
            )

        response = await asyncio.to_thread(_search)

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


    async def hybrid_search(self, query_text, top_k=5, alpha=0.5):

        def _search():

            response = self.collection.query.hybrid(
                query=query_text,
                alpha=alpha,
                limit=top_k,
                return_metadata=["score"]
            )

            results = []

            for obj in response.objects:

                score = getattr(obj.metadata, "score", None)

                results.append({
                    "id": str(obj.uuid),
                    "text": obj.properties.get("text", ""),
                    "metadata": obj.properties,
                    "score": score
                })

            return results

        return await asyncio.to_thread(_search)
    
    #this is for querying with pre-computed vectors, we can use dense_search for this but this is more explicit
    async def search(self, query_vector, top_k):

        def _search():

            response = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=["distance"]
            )

            results = []

            for obj in response.objects:

                results.append({
                    "id": str(obj.uuid),
                    "text": obj.properties.get("text", ""),
                    "score": obj.metadata.distance
                })

            return results