from fastapi import FastAPI, APIRouter
from app.retrieval.retriever import Retriever
from pydantic import BaseModel
from app.rag.simple_rag import SimpleRAG


from app.core.exception_handlers import (
    rag_exception_handler,
    generic_exception_handler,
)
from app.core.exceptions import RAGException


app = FastAPI()

app.add_exception_handler(RAGException, rag_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


router = APIRouter()
retriever = Retriever(top_k=5)
rag = SimpleRAG(top_k=5)


class QueryRequest(BaseModel):
    query: str


@router.post("/retrieve")
def retrieve(query: str):
    results = retriever.retrieve(query)
    return {"results": results}

@app.post("/ask")
def ask(request: QueryRequest):
    response = rag.run(request.query)
    return response

app.include_router(router)

