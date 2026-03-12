
import numpy
import torch
from sentence_transformers import SentenceTransformer



print("numpy:", numpy.__version__)
print("torch:", torch.__version__)

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
vec = model.encode("Apple revenue increased in 2024")

print("vector length:", len(vec))
