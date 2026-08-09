# Awesome LLM Apps — Findings for Personal KB Project

*Researched 2026-07-12 from github.com/Shubhamsaboo/awesome-llm-apps (118k stars, Apache-2.0)*

## Most Relevant Templates

### Tier 1 — Directly Applicable

| Template | Path | What it does | How it helps KB |
|----------|------|-------------|-----------------|
| **Knowledge Graph RAG** | `rag_tutorials/knowledge_graph_rag` | Builds knowledge graph from documents alongside vector retrieval | Connects to Hindsight's graph traversal; enhances cross-concept linking across lectures |
| **Agentic RAG (GPT-5)** | `rag_tutorials/agentic_rag_gpt5` | Agent decides when/how to retrieve, not just naive similarity search | Informs query routing — agent picks the right retrieval strategy per question |
| **Vision RAG** | `rag_tutorials/vision_rag` | RAG over images using Gemini | Relevant for processing lecture slide images/diagrams |
| **Chat with PDF** | `advanced_llm_apps/chat_with_X_tutorials/` | PDF ingestion → vector store → Q&A | Direct pattern for textbook/slide ingestion pipeline |
| **AI Teaching Agent Team** | `advanced_ai_agents/multi_agent_apps/agent_teams/` | Multi-agent system: quiz, explain, test knowledge | Directly applicable to Phase 3 (teach skill integration) |

### Tier 2 — Useful Patterns

| Template | Path | What it does | How it helps KB |
|----------|------|-------------|-----------------|
| **Memory-backed Apps (6 variants)** | `advanced_llm_apps/llm_apps_with_memory_tutorials/` | Conversations with persistent memory (mem0) | Pattern for Hindsight memory integration — replace mem0 with Hindsight |
| **RAG Failure Diagnostics** | `rag_tutorials/rag_failure_diagnostics_clinic` | Debugging tool for RAG pipelines | Useful for diagnosing retrieval quality during KB development |
| **Agentic RAG with Embedding Gemma** | `rag_tutorials/agentic_rag_embedding_gemma` | Uses Gemma embeddings for retrieval | Alternative embedding strategy to explore |
| **Voice RAG** | `voice_ai_agents/` | RAG over audio content | Relevant for Whisper→transcript→RAG pipeline |
| **Notion MCP Agent** | `mcp_ai_agents/` | Reads/writes Notion via MCP | Closest to Obsidian integration pattern |

### Tier 3 — Reference Only

| Template | Path | Notes |
|----------|------|-------|
| **Chat with YouTube** | `chat_with_X_tutorials/` | Transcript extraction pattern — could inform lecture audio processing |
| **Chat with ArXiv** | `chat_with_X_tutorials/` | Academic paper Q&A — useful for textbook processing |
| **GPT-OSS Critique Loop** | `advanced_llm_apps/` | Iterative improvement pattern — could inform note quality refinement |

## What's NOT in the Repo

- No Hindsight integration (they use mem0 for memory)
- No Obsidian integration
- No Whisper/STT pipeline (audio templates use hosted API endpoints)
- No template combines all 4 content types (slides + transcripts + textbooks + homework)
- No pgvector usage

## Recommended Starting Points

1. **Knowledge Graph RAG** — for retrieval architecture (graph + vector hybrid)
2. **Memory-backed Apps** — for memory integration pattern (swap mem0 → Hindsight)
3. **AI Teaching Agent Team** — for Phase 3 teach skill integration
4. **Vision RAG** — for processing lecture slide images

## Next Steps

- [ ] Clone `knowledge_graph_rag` template for closer inspection
- [ ] Evaluate if its graph construction approach complements Hindsight's
- [ ] Check `memory-backed_apps` for mem0 API patterns that map to Hindsight's retain/recall
- [ ] Review `AI Teaching Agent Team` for quiz/explain/test workflow
