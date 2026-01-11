# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

> **Note:** This project uses `uv` as the Python package manager (not pip/poetry).

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app runs on `http://localhost:8000` with API docs at `/docs`.

### Installing Dependencies
```bash
uv sync
```

### Running Commands with uv
```bash
uv run <command>    # Run any command in the virtual environment
uv run python ...   # Run Python scripts
uv run pytest       # Run tests (if added)
```

### Environment Setup
Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

This is a **tool-based RAG system** for querying course materials. The key design decision is that Claude *decides* when to search course content, rather than retrieving on every query.

### High-Level Flow

```
Frontend (JS) → FastAPI → RAGSystem → AIGenerator → Claude API
                                          ↑
                                    Tool Use Decision
                                          ↓
                              (optional) SearchTool → VectorStore → ChromaDB
```

### Component Responsibilities

| Component | File | Purpose |
|-----------|------|---------|
| **FastAPI** | `app.py` | REST endpoints (`/api/query`, `/api/courses`), serves static frontend |
| **RAGSystem** | `rag_system.py` | Orchestrator: wires together all components, manages sessions |
| **AIGenerator** | `ai_generator.py` | Claude API client with tool calling, handles tool execution loop |
| **SearchTool** | `search_tools.py` | Tool Claude calls to search; formats results for Claude to synthesize |
| **VectorStore** | `vector_store.py` | ChromaDB wrapper: semantic search, course name resolution, filtering |
| **DocumentProcessor** | `document_processor.py` | Parses course txt files into Course → Lesson → Chunk hierarchy |
| **SessionManager** | `session_manager.py` | In-memory conversation history (no persistence) |

### Tool-Based Design

The `search_course_content` tool is registered with ToolManager and passed to Claude via `tools` parameter. Claude decides autonomously whether to use it based on the query:

- **General knowledge** ("What is Python?") → Direct answer, no tool use
- **Course-specific** ("What does the MCP course cover?") → Tool use

### Document Format

Course documents in `docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [optional url]
[lesson content...]

Lesson 1: [next lesson title]
[lesson content...]
```

Each lesson is chunked by sentence with configurable overlap (default: 800 chars, 100 overlap).

### Vector Store Structure

ChromaDB has two collections:
- **`course_catalog`** - Course titles, instructor, links (for course name resolution via semantic search)
- **`course_content`** - Actual text chunks with metadata (course_title, lesson_number, chunk_index)

Course name matching is semantic: "MCP" matches "MCP: Build Rich-Context AI Apps".

### Session Management

Sessions are stored in-memory only (no persistence). Each session tracks up to `MAX_HISTORY` message pairs (default: 2). Session IDs are simple incrementing counters (`session_1`, `session_2`).

### Configuration

All settings in `config.py`:
- `ANTHROPIC_MODEL` - Claude model to use
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - Text chunking parameters
- `MAX_RESULTS` - Search results returned (default: 5)
- `MAX_HISTORY` - Conversation turns remembered (default: 2)

### Frontend

Vanilla JavaScript in `frontend/`:
- `script.js` - Chat UI, API calls, markdown rendering via `marked.js`
- `index.html` - Main interface
- `style.css` - Styling

Static files are served from `backend/app.py` via `StaticFiles` mount.

### Startup Behavior

On startup, `app.py` loads all documents from `../docs` into ChromaDB automatically. Existing courses are skipped (checked by title). To rebuild, delete `chroma_db/` directory.
