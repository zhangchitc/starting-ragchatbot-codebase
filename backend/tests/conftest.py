"""
Pytest fixtures for RAG system testing.
"""
import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults, VectorStore
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem


# Sample test data
SAMPLE_COURSE_TITLE = "MCP: Build Rich-Context AI Apps"
SAMPLE_COURSE_LINK = "https://example.com/mcp"
SAMPLE_INSTRUCTOR = "Test Instructor"


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test-api-key-123"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return SearchResults(
        documents=[
            "MCP is a protocol that enables AI to interact with external tools and data sources.",
            "The MCP server provides context and capabilities to AI models."
        ],
        metadata=[
            {
                "course_title": SAMPLE_COURSE_TITLE,
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": SAMPLE_COURSE_TITLE,
                "lesson_number": 2,
                "chunk_index": 1
            }
        ],
        distances=[0.23, 0.31],
        error=None
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing."""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def error_search_results():
    """Create search results with error for testing."""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Search error: ChromaDB connection failed"
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    mock_store = Mock(spec=VectorStore)

    # Mock search method - returns sample results by default
    mock_store.search = Mock(return_value=SearchResults(
        documents=["Test content about MCP"],
        metadata=[{"course_title": SAMPLE_COURSE_TITLE, "lesson_number": 1, "chunk_index": 0}],
        distances=[0.1],
        error=None
    ))

    # Mock _resolve_course_name for semantic matching
    mock_store._resolve_course_name = Mock(return_value=SAMPLE_COURSE_TITLE)

    # Mock get_course_link and get_lesson_link
    mock_store.get_course_link = Mock(return_value=SAMPLE_COURSE_LINK)
    mock_store.get_lesson_link = Mock(return_value=f"{SAMPLE_COURSE_LINK}/lesson1")

    # Mock get_all_courses_metadata
    mock_store.get_all_courses_metadata = Mock(return_value=[
        {
            "title": SAMPLE_COURSE_TITLE,
            "instructor": SAMPLE_INSTRUCTOR,
            "course_link": SAMPLE_COURSE_LINK,
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction to MCP", "lesson_link": f"{SAMPLE_COURSE_LINK}/lesson0"},
                {"lesson_number": 1, "lesson_title": "Building MCP Servers", "lesson_link": f"{SAMPLE_COURSE_LINK}/lesson1"},
                {"lesson_number": 2, "lesson_title": "Advanced MCP Features", "lesson_link": f"{SAMPLE_COURSE_LINK}/lesson2"},
            ],
            "lesson_count": 3
        }
    ])

    # Mock course_catalog for CourseOutlineTool
    mock_catalog = Mock()
    mock_catalog.get = Mock(return_value={
        'metadatas': [{
            'title': SAMPLE_COURSE_TITLE,
            'course_link': SAMPLE_COURSE_LINK,
            'lessons_json': '[{"lesson_number": 0, "lesson_title": "Intro", "lesson_link": "link0"}, {"lesson_number": 1, "lesson_title": "Advanced", "lesson_link": "link1"}]',
            'lesson_count': 2
        }]
    })
    mock_store.course_catalog = mock_catalog

    return mock_store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool instance with mock vector store."""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create a CourseOutlineTool instance with mock vector store."""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create a ToolManager with registered tools."""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing."""
    mock_client = MagicMock()

    # Mock the messages.create method
    mock_response = MagicMock()
    mock_response.stop_reason = "stop"
    mock_content = MagicMock()
    mock_content.text = "This is a test response about MCP."
    mock_response.content = [mock_content]

    mock_client.messages.create = Mock(return_value=mock_response)
    return mock_client


@pytest.fixture
def ai_generator_with_mock(mock_anthropic_client, mock_config):
    """Create an AIGenerator with mocked Anthropic client."""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        generator.client = mock_anthropic_client
        return generator


@pytest.fixture
def tool_use_response():
    """Create a mock Anthropic response that triggers tool use."""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"

    # Create mock tool_use block
    mock_tool_use = MagicMock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.id = "toolu_123"
    mock_tool_use.name = "search_course_content"
    mock_tool_use.input = {"query": "What is MCP?"}

    mock_response.content = [mock_tool_use]
    return mock_response


@pytest.fixture
def ai_generator_with_tool_use(mock_anthropic_client, mock_config, tool_use_response):
    """Create an AIGenerator that will return tool_use response."""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        generator.client = mock_anthropic_client
        # First call returns tool_use, second returns final response
        mock_anthropic_client.messages.create = Mock(side_effect=[
            tool_use_response,
            MagicMock(
                stop_reason="stop",
                content=[MagicMock(text="Based on the search results, MCP is...")]
            )
        ])
        return generator


@pytest.fixture
def sample_course_metadata():
    """Sample course metadata for testing."""
    return {
        "title": SAMPLE_COURSE_TITLE,
        "instructor": SAMPLE_INSTRUCTOR,
        "course_link": SAMPLE_COURSE_LINK,
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "link0"},
            {"lesson_number": 1, "lesson_title": "Advanced Topics", "lesson_link": "link1"},
        ],
        "lesson_count": 2
    }


# ============================================================================
# API Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """Create a fully mocked RAGSystem for API testing."""
    mock_system = Mock()
    mock_system.session_manager = Mock()

    # Mock session creation
    mock_system.session_manager.create_session = Mock(return_value="test_session_123")
    mock_system.session_manager.get_conversation_history = Mock(return_value=None)
    mock_system.session_manager.add_exchange = Mock()

    # Mock query method - returns tuple of (response, sources)
    mock_system.query = Mock(return_value=(
        "MCP is a protocol that enables AI to interact with external tools and data sources.",
        [{"text": "MCP Course", "url": "https://example.com/mcp"}]
    ))

    # Mock get_course_analytics
    mock_system.get_course_analytics = Mock(return_value={
        "total_courses": 3,
        "course_titles": [
            "MCP: Build Rich-Context AI Apps",
            "Building AI Assistants with Claude",
            "Advanced Prompt Engineering"
        ]
    })

    return mock_system


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.

    This fixture creates a minimal FastAPI app for testing API endpoints
    without the static file middleware that requires actual frontend files.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Pydantic models (matching app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceInfo(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceInfo]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Create test app
    app = FastAPI(title="Test RAG System API")

    # Add middleware (matching app.py)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Store the mock RAG system in app state for access in endpoints
    app.state.rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources."""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics."""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint - API health check."""
        return {"status": "ok", "message": "RAG System API is running"}

    return app


@pytest.fixture
def client(test_app):
    """
    Create an AsyncHTTPClient for testing FastAPI endpoints.

    Uses TestClient from FastAPI's Starlette for synchronous testing.
    """
    from fastapi.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request payload."""
    return {"query": "What is MCP?"}


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request payload with session_id."""
    return {
        "query": "What is MCP?",
        "session_id": "existing_session_456"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response for assertion testing."""
    return {
        "answer": "MCP is a protocol that enables AI to interact with external tools and data sources.",
        "sources": [
            {"text": "MCP Course", "url": "https://example.com/mcp"}
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_courses_response():
    """Sample courses response for assertion testing."""
    return {
        "total_courses": 3,
        "course_titles": [
            "MCP: Build Rich-Context AI Apps",
            "Building AI Assistants with Claude",
            "Advanced Prompt Engineering"
        ]
    }


@pytest.fixture
def error_mock_rag_system():
    """RAG system mock that raises errors for testing error handling."""
    mock_system = Mock()
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session = Mock(return_value="error_session")

    # Mock query that raises an exception
    mock_system.query = Mock(side_effect=Exception("API key invalid"))

    # Mock get_course_analytics that raises an exception
    mock_system.get_course_analytics = Mock(
        side_effect=Exception("Database connection failed")
    )

    return mock_system


@pytest.fixture
def error_app(error_mock_rag_system):
    """
    Create a test FastAPI app that returns errors for testing error handling.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceInfo(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceInfo]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    app = FastAPI(title="Test RAG System API - Error Mode")

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or error_mock_rag_system.session_manager.create_session()
            answer, sources = error_mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = error_mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "RAG System API is running"}

    return app


@pytest.fixture
def error_client(error_app):
    """Test client for error testing."""
    from fastapi.testclient import TestClient
    return TestClient(error_app)


# ============================================================================
# pytest configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests (may require real dependencies)")
    config.addinivalue_line("markers", "unit: marks tests as unit tests (isolated, mocked)")
    config.addinivalue_line("markers", "slow: marks tests as slow (may require network or heavy computation)")
    config.addinivalue_line("markers", "api: marks tests as API endpoint tests")
