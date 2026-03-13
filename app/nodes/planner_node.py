import json
from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def create_planner_node(llm, tool_registry):

    async def planner_node(state: AgentState):

        log_node_start("planner", state)

        query = state.query
        tools_description = tool_registry.format_for_prompt()

        prompt = f"""
            You are an AI planning agent.

            Available tools:

            {tools_description}

            Create a plan to answer the user query.

            Return a JSON array of steps.

            Each step must contain:
            - tool
            - input (object with parameters)

            Example:

            [
            {{
            "tool": "vector_search",
            "input": {{"query": "Apple revenue growth"}}
            }},
            {{
            "tool": "web_search",
            "input": {{"query": "Apple stock price today"}}
            }},
            {{
            "tool": "generate"
            }}
            ]

            Rules:
            - Always end with "generate"
            - Use vector_search for internal documents
            - Use web_search for current information
            - Return ONLY JSON

            Query:
            {query}
            """

        response = await llm.ainvoke(prompt)

        try:
            plan = json.loads(response.content.strip())
        except Exception:
            plan = [
                {"tool": "vector_search", "input": {"query": query}},
                {"tool": "generate"}
            ]

        print("\n🧠 Planner Output")
        print("Query:", query)
        print("Plan :", plan)
        print()

        log_node_end("planner", state)

        return {
            "plan": plan
        }

    return planner_node