from app.agents.state import AgentState


def route_after_router(state: AgentState):

    plan = state.plan
    step = state.current_step

    if step >= len(plan):
        return "validator"

    action = plan[step]

    mapping = {
        "vector_search": "retriever",
        "web_search": "executor",
        "generate": "generator"
    }

    return mapping.get(action, "generator")