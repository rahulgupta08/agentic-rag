from typing import  Dict, List, Optional, Any
from pydantic import BaseModel

class AgentState(BaseModel):
    """
    Represents the state of an agent, including its name, description, and any additional information.
    """
    query: str

    retrieved_docs: Optional[List[Dict[str, Any]]] = None

    context: Optional[str] = None

    tool_name: Optional[str] = None
    tool_output: Optional[str] = None

    is_valid: Optional[bool] = None
    retry_count: int = 0

    answer: Optional[str] = None

    metadata: Dict[str, Any] = {}