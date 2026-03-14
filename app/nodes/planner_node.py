import json
from app.agents.state import AgentState
from app.core.observability import metrics



def create_planner_node(llm, tool_registry):

    async def planner_node(state: AgentState):

        

        query = state.query
        tools_description = tool_registry.format_for_prompt()
        tool_history = ""

        if state.tool_results:

            history_lines = []

            for result in state.tool_results:
                tool_name = result.get("tool")
                output = result.get("output")

                if not output:
                    history_lines.append(f"{tool_name} → returned no results")
                else:
                    history_lines.append(f"{tool_name} → returned results")

            tool_history = "\nPrevious tool attempts:\n" + "\n".join(history_lines)

        prompt = f"""
        You are an AI planning agent responsible for deciding how to answer a user question using available tools.

        Your task is to create a step-by-step plan that uses the provided tools to gather information and produce the final answer.

        You do NOT execute tools yourself. You only create the plan.

        --------------------------------------------------
        AVAILABLE TOOLS
        --------------------------------------------------

        {tools_description}

        Each tool has a specific capability. Choose the tools that best help answer the user query.

        --------------------------------------------------
        PREVIOUS TOOL ATTEMPTS
        --------------------------------------------------

        {tool_history if tool_history else "No previous attempts."}

        If a tool previously returned no useful results, avoid repeating it unless absolutely necessary.

        --------------------------------------------------
        PLANNING STRATEGY
        --------------------------------------------------

        Follow these guidelines:

        1. Understand what the user is asking.
        2. Decide which information sources are required.
        3. Select appropriate tools to retrieve that information.
        4. Use multiple tools if necessary.
        5. After gathering information, use the "generate" step to produce the final answer.

        Examples:

        Example 1 — Internal document question:

        Query:
        "What was Apple's revenue growth in 2023?"

        Plan:
        [
        {{
            "tool": "vector_search",
            "input": {{"query": "Apple revenue growth 2023"}}
        }},
        {{
            "tool": "generate"
        }}
        ]

        Example 2 — Current information:

        Query:
        "What is Apple's stock price today?"

        Plan:
        [
        {{
            "tool": "web_search",
            "input": {{"query": "Apple current stock price"}}
        }},
        {{
            "tool": "generate"
        }}
        ]

        Example 3 — Multi-source query:

        Query:
        "What was Apple's revenue in the 2023 10K and what is its current stock price?"

        Plan:
        [
        {{
            "tool": "vector_search",
            "input": {{"query": "Apple revenue 2023 10K"}}
        }},
        {{
            "tool": "web_search",
            "input": {{"query": "Apple stock price today"}}
        }},
        {{
            "tool": "generate"
        }}
        ]

        --------------------------------------------------
        IMPORTANT RULES
        --------------------------------------------------

        - Always return a valid JSON array.
        - Each step must contain a "tool".
        - Include an "input" object when the tool requires parameters.
        - Always end the plan with the "generate" tool.
        - Do not repeat tools that previously failed unless necessary.
        - Do not include explanations or text outside the JSON.

        --------------------------------------------------
        USER QUERY
        --------------------------------------------------

        {query}

        --------------------------------------------------
        OUTPUT FORMAT
        --------------------------------------------------

        Return ONLY the JSON plan.
        """

        response = await llm.ainvoke(prompt,temperature = 0)

        try:
            plan = json.loads(response.content.strip())
        except Exception:
            plan = [
                {"tool": "vector_search", "input": {"query": query}},
                {"tool": "generate"}
            ]

        metrics.log_plan(plan)

        return {
            "plan": plan
        }

    return planner_node