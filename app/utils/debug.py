def log_node_start(node_name, state):
    print(f"\n NODE START: {node_name}")
    print("State keys:", list(state.keys()) if isinstance(state, dict) else state.model_dump().keys())


def log_node_end(node_name, state):
    print(f"\n NODE END: {node_name}")