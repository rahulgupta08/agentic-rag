from app.services.sqlite_service import sqlite_service


def memory_writer_node(state):

    sqlite_service.save_message(
        session_id=state.session_id,
        role="user",
        message=state.query,
    )

    sqlite_service.save_message(
        session_id=state.session_id,
        role="assistant",
        message=state.answer,
    )

    return {}