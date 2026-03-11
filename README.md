# Financial RAG System

This project implements a Retrieval Augmented Generation (RAG) system
for financial document analysis.

Current Version:
Baseline RAG pipeline before Agentic upgrade.

Planned upgrades:
- LangGraph agent architecture
- MCP tool integration
- Memory layer
- LangSmith observability
- RAGAS evaluation

Architecture:

User Query
   ↓
Embedding
   ↓
Vector Search
   ↓
Document Retrieval
   ↓
LLM
   ↓
Answer