import asyncio
import time
from typing import Any, Optional
from app.mcp.client.mcp_client import MCPClient
from app.core.exceptions import MCPError

class CircuitBreaker:
    """
    A circuit breaker implementation for MCP client calls.
    
    This prevents cascading failures when external services are unavailable.
    """
    
    def __init__(self, client: MCPClient, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 reset_timeout: float = 30.0):
        """
        Initialize the circuit breaker.
        
        Args:
            client: The underlying MCP client
            failure_threshold: Number of failures before opening circuit
            timeout: Timeout for calls (seconds)
            reset_timeout: Time to wait before attempting to reset circuit (seconds)
        """
        self.client = client
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        
        # Circuit state tracking
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_attempt_time: Optional[float] = None
        
    async def call_tool(self, tool_name: str, tool_input: dict) -> Any:
        """
        Call a tool with circuit breaker protection.
        
        Args:
            tool_name: Name of the tool to call
            tool_input: Input parameters for the tool
            
        Returns:
            The result from the tool call
            
        Raises:
            MCPError: If the circuit is open or tool call fails
        """
        # Check if circuit is open
        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.reset_timeout:
                # Move to HALF_OPEN state to test if service is back
                self.state = "HALF_OPEN"
                return await self._attempt_call(tool_name, tool_input)
            else:
                # Circuit is still open, reject the call
                raise MCPError("Circuit breaker is OPEN. Service temporarily unavailable.")
        
        # Circuit is CLOSED or HALF_OPEN, attempt the call
        try:
            result = await self._attempt_call(tool_name, tool_input)
            # Reset failure count on success
            self.failure_count = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            # Handle failure
            self._record_failure()
            raise e
    
    async def _attempt_call(self, tool_name: str, tool_input: dict) -> Any:
        """
        Attempt the actual tool call with timeout.
        """
        try:
            # Add timeout to the call
            async with asyncio.timeout(self.timeout):
                result = await self.client.call_tool(tool_name, tool_input)
                return result
        except asyncio.TimeoutError:
            raise MCPError(f"Timeout calling tool {tool_name}")
        except Exception as e:
            raise MCPError(f"Error calling tool {tool_name}: {str(e)}")
    
    def _record_failure(self):
        """
        Record a failure and update circuit state.
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.last_attempt_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.failure_count = 0  # Reset count after opening circuit

# Create a circuit breaker instance for the application
# This will be initialized in the factory
mcp_circuit_breaker: Optional[CircuitBreaker] = None

def get_mcp_circuit_breaker() -> CircuitBreaker:
    """
    Get the global MCP circuit breaker instance.
    """
    global mcp_circuit_breaker
    if mcp_circuit_breaker is None:
        raise RuntimeError("MCP circuit breaker not initialized")
    return mcp_circuit_breaker

def initialize_mcp_circuit_breaker(client: MCPClient) -> CircuitBreaker:
    """
    Initialize the global MCP circuit breaker with a client.
    """
    global mcp_circuit_breaker
    mcp_circuit_breaker = CircuitBreaker(client)
    return mcp_circuit_breaker