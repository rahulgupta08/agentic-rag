from app.rag.advanced_rag import AdvancedRAG

rag = AdvancedRAG(similarity_threshold=0.5)

response = rag.ask("when is Independence day celebrated in India?", "hybrid")

print(response["confidence"])
print(response["answer"])