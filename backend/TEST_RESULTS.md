# Test Results and Root Cause Analysis

## Test Summary
- **Total tests written**: 83
- **Tests passed**: 83
- **Test coverage**:
  - `test_search_tools.py`: 27 tests
  - `test_ai_generator.py`: 19 tests
  - `test_vector_store.py`: 23 tests
  - `test_rag_system.py`: 14 tests

## Root Cause Identified: Missing API Key

**The primary cause of "query failed" is: `ANTHROPIC_API_KEY` is not set.**

### Evidence
1. No `.env` file exists in `backend/`
2. No `.env.example` file exists as a template
3. `config.py` loads from `.env` but defaults to empty string if not found

### Impact
When `ANTHROPIC_API_KEY` is empty or invalid:
- `anthropic.Anthropic()` API calls fail with 401 Unauthorized
- The exception propagates through `AIGenerator.generate_response()`
- Caught by `app.py` error handler → HTTP 500 → Frontend shows "Query failed"

## Secondary Issues Found by Tests

### 1. Uncaught VectorStore Exceptions
**Location**: `vector_store.py:66-100`

The `search()` method wraps ChromaDB calls in try/except and returns `SearchResults` with an error message. However, if an exception occurs **outside** the try block (e.g., during query parameter preparation), it propagates uncaught.

**Current code**:
```python
def search(self, query: str, course_name: Optional[str] = None, ...):
    course_title = None
    if course_name:
        course_title = self._resolve_course_name(course_name)  # Not in try/except!
        if not course_title:
            return SearchResults.empty(f"No course found matching '{course_name}'")
    ...
    try:
        results = self.course_content.query(...)  # Only this is wrapped
        return SearchResults.from_chroma(results)
    except Exception as e:
        return SearchResults.empty(f"Search error: {str(e)}")
```

### 2. Uncaught Tool Execution Exceptions
**Location**: `search_tools.py:52-86`

The `CourseSearchTool.execute()` method calls `self.store.search()` directly. If `store.search()` raises an exception (which it can, per issue #1), it propagates up.

### 3. Poor Error Messages in Frontend
**Location**: `app.py:78-79`, `frontend/script.js:79`

The error handler just returns `str(e)`, which could be:
- "401 Unauthorized" (not user-friendly)
- "Connection timeout" (doesn't say which service)
- Internal Python tracebacks

## Recommended Fixes

### Fix 1: Add API Key Validation (Immediate Priority)

Create `backend/.env.example`:
```bash
# Anthropic API Settings
ANTHROPIC_API_KEY=your_api_key_here

# Optional: Override default model
# ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

Add startup validation to `config.py`:
```python
@dataclass
class Config:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    def __post_init__(self):
        if not self.ANTHROPIC_API_KEY:
            import warnings
            warnings.warn(
                "ANTHROPIC_API_KEY is not set! The application will not work. "
                "Create a .env file with your API key."
            )
```

### Fix 2: Improve Error Handling in VectorStore

Move the `_resolve_course_name` call inside the try block or wrap it separately:
```python
def search(self, query: str, course_name: Optional[str] = None, ...):
    try:
        course_title = None
        if course_name:
            course_title = self._resolve_course_name(course_name)
            if not course_title:
                return SearchResults.empty(f"No course found matching '{course_name}'")

        filter_dict = self._build_filter(course_title, lesson_number)

        results = self.course_content.query(
            query_texts=[query],
            n_results=search_limit,
            where=filter_dict
        )
        return SearchResults.from_chroma(results)
    except Exception as e:
        return SearchResults.empty(f"Search error: {str(e)}")
```

### Fix 3: Better Error Messages in API Response

Update `app.py`:
```python
@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        session_id = request.session_id or rag_system.session_manager.create_session()
        answer, sources = rag_system.query(request.query, session_id)
        return QueryResponse(answer=answer, sources=sources, session_id=session_id)
    except Exception as e:
        import traceback
        error_detail = str(e)

        # Add context for common errors
        if "401" in error_detail or "Unauthorized" in error_detail:
            error_detail = "API authentication failed. Please check your ANTHROPIC_API_KEY."
        elif "timeout" in error_detail.lower():
            error_detail = "Request timed out. Please try again."
        elif "connection" in error_detail.lower():
            error_detail = "Unable to connect to the service. Please check your internet connection."

        # Log the full error for debugging
        print(f"Query error: {error_detail}")
        print(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_detail)
```

### Fix 4: Add Health Check Endpoint

Add to `app.py`:
```python
@app.get("/api/health")
async def health_check():
    """Health check endpoint that verifies all services are working."""
    health_status = {
        "status": "healthy",
        "services": {}
    }

    # Check API key
    if config.ANTHROPIC_API_KEY:
        health_status["services"]["anthropic_api"] = "configured"
    else:
        health_status["services"]["anthropic_api"] = "missing"
        health_status["status"] = "unhealthy"

    # Check ChromaDB
    try:
        course_count = rag_system.vector_store.get_course_count()
        health_status["services"]["chromadb"] = f"ok ({course_count} courses)"
    except Exception as e:
        health_status["services"]["chromadb"] = f"error: {str(e)}"

    return health_status
```

## Implementation Priority

1. **Do first**: Create `.env` file with `ANTHROPIC_API_KEY` to fix immediate issue
2. **Do second**: Add `.env.example` and startup validation
3. **Do third**: Improve error handling in `vector_store.py`
4. **Do fourth**: Better error messages in `app.py`
5. **Do fifth**: Add health check endpoint
