import weaviate

# Connect to the local instance
client = weaviate.connect_to_local()

print(client.collections.list_all())
#result = client.collections.get("financial_documents")
#print(result)

if client.is_ready():
    print("Weaviate is ready!")
    #print(client.get_meta())
    
client.close()

#docker compose up -d
#docker compose stop
#docker compose down