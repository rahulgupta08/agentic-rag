from app.services.rag_service import RAGService
from scripts.test_rag_pipeline import build_rag_service


def register_vector_search_tool(mcp):

    rag_service = build_rag_service()

    @mcp.tool()
    async def vector_search(query: str):

        documents = await rag_service.retriever.retrieve(query)

        results = []

        for doc in documents:
            if hasattr(doc, "page_content"):
                results.append(doc.page_content)
            else:
                results.append(str(doc))

        return results