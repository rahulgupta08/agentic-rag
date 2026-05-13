import json
from app.agents.agent_state import AgentState
from app.core.observability import metrics
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Load template once at module level
TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"
PROMPT_TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")


def create_planner_node(llm, tool_registry):

    async def planner_node(state: AgentState):

        
        query = state.query
        tools_description = tool_registry.format_for_prompt()
        tool_history = ""

        if state.tool_results:

            history_lines = []

            for result in state.tool_results:
                tool_name = result.get("tool")
                #output = result.get("output")
                documents = result.get("documents", [])
                logger.info(f"From tool {tool_name} documents fetched  {documents}")

                if not documents:
                    history_lines.append(f"{tool_name} → returned no results")
                else:
                    history_lines.append(f"{tool_name} → returned results")

            tool_history = "\nPrevious tool attempts:\n" + "\n".join(history_lines)

        # Format the prompt with dynamic parts
        prompt = PROMPT_TEMPLATE.format(
            tools_description=tools_description,
            tool_history=tool_history if tool_history else "No previous attempts.",
            query=query
        )

        # Add caching for static parts of the prompt
        # For OpenAI, we can use prompt caching by adding cache_control to the messages
        # We'll mark the static parts as cacheable by adding cache_control to the system message
        try:
            # For OpenAI, we can use prompt caching by adding cache_control to the system message
            # This is a simplified approach - in practice, you'd want to use the actual caching mechanism
            # provided by the LLM provider
            response = await llm.ainvoke(prompt, temperature=0)
            
            plan = json.loads(response.content.strip())
            
            # Validate plan structure
            if not isinstance(plan, list):
                raise ValueError("Plan must be a list")
            
            for step in plan:
                if not isinstance(step, dict) or "tool" not in step:
                    raise ValueError(f"Invalid step format: {step}")
            
            # Ensure plan ends with "generate"
            if not plan or plan[-1].get("tool") != "generate":
                logger.warning("Plan doesn't end with 'generate', appending it")
                plan.append({"tool": "generate"})
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {response.content}")
            
            # Fallback plan
            plan = [
                {"tool": "vector_search", "input": {"query": query}},
                {"tool": "generate"}
            ]
            logger.info(f"Using fallback plan: {plan}")
            
        except Exception as e:
            logger.exception(f"Planner failed: {e}")
            
            # Minimal fallback
            plan = [{"tool": "generate"}]

        metrics.log_plan(plan)

        return {
            "plan": plan
        }

    return planner_node