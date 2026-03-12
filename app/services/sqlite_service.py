import sqlite3
from pathlib import Path
from typing import List, Dict


DB_PATH = Path("conversation_data/agent_memory.db")


class SQLiteService:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_db(self):
        """Create table if it does not exist"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()
        conn.close()

    def save_message(self, session_id: str, role: str, message: str):
        """Save a conversation message"""

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations (session_id, role, message)
            VALUES (?, ?, ?)
            """,
            (session_id, role, message),
        )

        conn.commit()
        conn.close()

    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[Dict]:

        """Fetch recent conversation history"""

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT role, message
            FROM conversations
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        )

        rows = cursor.fetchall()

        conn.close()

        rows.reverse()

        return [
            {"role": role, "message": message}
            for role, message in rows
        ]


sqlite_service = SQLiteService()