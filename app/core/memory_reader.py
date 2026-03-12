from app.services.sqlite_service import sqlite_service


def memory_reader_node(state):

    history = sqlite_service.get_recent_messages(
        session_id=state.session_id,
        limit=5,
    )
    state.conversation_history = history
    return state