from typing import List, Optional, Any
from pydantic import BaseModel, Field

class AgentState(BaseModel):

    # Session
    # loading memory from SQLite
    # storing conversation history
    session_id: Optional[str] = None

    # User question
    query: str

    # Query classification, either a RAG query, web search or hybrid query
    query_type: Optional[str] = None

    # Planner execution
    plan: List[str] = Field(default_factory=list)
    # current step in the plan execution. Tracks execution progess
    current_step: int = 0

    context: Optional[str] = None

    # Retrieval results from RAG service
    retrieved_docs: Optional[List[Any]] = None

    # Tool execution results , web search
    web_results: Optional[List[Any]] = None

    # Generated answer
    answer: Optional[str] = None

    # Validation score will be used by RAGAS
    validation_score: Optional[float] = None
    #self correction loop
    needs_retry: Optional[bool] = None

    conversation_history: Optional[List[Any]] = None