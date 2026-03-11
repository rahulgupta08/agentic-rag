import weaviate

# Connect to the local instance
client = weaviate.connect_to_local()

if client.is_ready():
    print("Weaviate is ready!")
    print(client.get_meta())
    
client.close()

#docker compose up -d
#docker compose stop
#docker compose down