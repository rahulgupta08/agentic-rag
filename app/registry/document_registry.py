import sqlite3
from pathlib import Path


class DocumentRegistry:

    def __init__(self, db_path="data/registry.db"):

        Path("data").mkdir(exist_ok=True)

        self.conn = sqlite3.connect(db_path)

        self._initialize_schema()


    def _initialize_schema(self):

        with open("app/registry/schema.sql") as f:
            self.conn.executescript(f.read())


    # ----------------------------
    # Insert document
    # ----------------------------

    def register_document(self,
                          document_id,
                          file_path,
                          file_hash,
                          collection,
                          embedding_model):

        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (document_id, file_path, file_hash, collection, embedding_model, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                file_path,
                file_hash,
                collection,
                embedding_model,
                "processing"
            )
        )

        self.conn.commit()


    # ----------------------------
    # Update status
    # ----------------------------

    def update_status(self, document_id, status, chunk_count=None):

        self.conn.execute(
            """
            UPDATE documents
            SET status = ?, chunk_count = ?, updated_at = CURRENT_TIMESTAMP
            WHERE document_id = ?
            """,
            (status, chunk_count, document_id)
        )

        self.conn.commit()


    # ----------------------------
    # Check if document exists
    # ----------------------------

    def document_exists(self, file_hash):

        cursor = self.conn.execute(
            """
            SELECT document_id FROM documents
            WHERE file_hash = ?
            """,
            (file_hash,)
        )

        result = cursor.fetchone()

        return result is not None