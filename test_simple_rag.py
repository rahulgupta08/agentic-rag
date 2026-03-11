from app.rag.simple_rag import SimpleRAG

def main():

    rag = SimpleRAG()

    query = "What risk factors does Apple mention in the 10-K?"

    print("\nQUERY:")
    print(query)

    response = rag.ask(question=query, top_k=10)

    print("\nANSWER:")
    print(response["answer"])

    print("\nRETRIEVED DOCUMENTS:")

    # for i, doc in enumerate(response["retrieved_docs"], 1):
    #     print(f"\nDocument {i}")
    #     print("ID:", doc.get("id"))
    #     print("Score:", doc.get("score"))
    #     print("Preview:", doc.get("text", "")[:50])


if __name__ == "__main__":
    main()