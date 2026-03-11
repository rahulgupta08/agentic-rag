CREATE TABLE IF NOT EXISTS documents (

    document_id TEXT PRIMARY KEY,

    file_path TEXT NOT NULL,

    file_hash TEXT NOT NULL,

    collection TEXT NOT NULL,

    embedding_model TEXT NOT NULL,

    status TEXT NOT NULL,

    chunk_count INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);