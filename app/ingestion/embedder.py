
from openai import OpenAI
from app.config import EMBEDDING_MODEL
import logging
from dotenv import load_dotenv
from app.config import  OPENAI_API_KEY
from sentence_transformers import SentenceTransformer


load_dotenv()

logger = logging.getLogger(__name__)
# client = OpenAI(api_key=OPENAI_API_KEY)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")




def embed_text(text: str):

    text = f"Represent this sentence for searching relevant passages: {text}"

    embedding = model.encode(text, normalize_embeddings=True)

    return embedding.tolist()