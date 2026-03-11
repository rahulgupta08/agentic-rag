import weaviate
from weaviate.classes.query import Filter

client = weaviate.connect_to_local(
    host="localhost",
    port=8080
)

collection = client.collections.get("financial_documents")

# print("Total vectors:", collection.aggregate.over_all().total_count)

# objects = collection.query.fetch_objects(limit=5)

# print("\nSample objects:\n")

# for obj in objects.objects:
#     print(obj.properties)

response = collection.query.fetch_objects(
    filters=(
        Filter.by_property("ticker").equal("AAPL") &
        Filter.by_property("year").equal("2024")
    ),
    limit=10
)


for obj in response.objects:
    print(obj.properties)

print("Total objects:")
print(collection.aggregate.over_all().total_count)
# FOr googl 2025 .. Inserted vectors: 196 , total object count: 196
# For googl 2024 .. Inserted vectors: 208 , total object count: 196 + 208 = 404 
# For googl 2023 .. Inserted vectors: 162 , total object count: 404 + 162 = 566 --- so total google objects = 566

# FOr AAPL 2025 .. Inserted vectors: 104 , total object count: 566 + 104 = 670
# For AAPL 2024 .. Inserted vectors: 102 , total object count: 670 + 102 = 772
# For AAPL 2023 .. Inserted vectors: 99 , total object count: 772 + 99 = 871 
# Google total = 566
# Apple total = 871 - 566 = 305

