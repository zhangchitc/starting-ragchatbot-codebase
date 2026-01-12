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
            "The MCP server provides context and capabilities to AI models.",
        ],
        metadata=[
            {"course_title": SAMPLE_COURSE_TITLE, "lesson_number": 1, "chunk_index": 0},
            {"course_title": SAMPLE_COURSE_TITLE, "lesson_number": 2, "chunk_index": 1},
        ],
        distances=[0.23, 0.31],
        error=None,
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing."""
    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def error_search_results():
    """Create search results with error for testing."""
    return SearchResults(
        documents=[], metadata=[], distances=[], error="Search error: ChromaDB connection failed"
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    mock_store = Mock(spec=VectorStore)

    # Mock search method - returns sample results by default
    mock_store.search = Mock(
        return_value=SearchResults(
            documents=["Test content about MCP"],
            metadata=[{"course_title": SAMPLE_COURSE_TITLE, "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
            error=None,
        )
    )

    # Mock _resolve_course_name for semantic matching
    mock_store._resolve_course_name = Mock(return_value=SAMPLE_COURSE_TITLE)

    # Mock get_course_link and get_lesson_link
    mock_store.get_course_link = Mock(return_value=SAMPLE_COURSE_LINK)
    mock_store.get_lesson_link = Mock(return_value=f"{SAMPLE_COURSE_LINK}/lesson1")

    # Mock get_all_courses_metadata
    mock_store.get_all_courses_metadata = Mock(
        return_value=[
            {
                "title": SAMPLE_COURSE_TITLE,
                "instructor": SAMPLE_INSTRUCTOR,
                "course_link": SAMPLE_COURSE_LINK,
                "lessons": [
                    {
                        "lesson_number": 0,
                        "lesson_title": "Introduction to MCP",
                        "lesson_link": f"{SAMPLE_COURSE_LINK}/lesson0",
                    },
                    {
                        "lesson_number": 1,
                        "lesson_title": "Building MCP Servers",
                        "lesson_link": f"{SAMPLE_COURSE_LINK}/lesson1",
                    },
                    {
                        "lesson_number": 2,
                        "lesson_title": "Advanced MCP Features",
                        "lesson_link": f"{SAMPLE_COURSE_LINK}/lesson2",
                    },
                ],
                "lesson_count": 3,
            }
        ]
    )

    # Mock course_catalog for CourseOutlineTool
    mock_catalog = Mock()
    mock_catalog.get = Mock(
        return_value={
            "metadatas": [
                {
                    "title": SAMPLE_COURSE_TITLE,
                    "course_link": SAMPLE_COURSE_LINK,
                    "lessons_json": '[{"lesson_number": 0, "lesson_title": "Intro", "lesson_link": "link0"}, {"lesson_number": 1, "lesson_title": "Advanced", "lesson_link": "link1"}]',
                    "lesson_count": 2,
                }
            ]
        }
    )
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
    with patch("ai_generator.anthropic.Anthropic", return_value=mock_anthropic_client):
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
    with patch("ai_generator.anthropic.Anthropic", return_value=mock_anthropic_client):
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        generator.client = mock_anthropic_client
        # First call returns tool_use, second returns final response
        mock_anthropic_client.messages.create = Mock(
            side_effect=[
                tool_use_response,
                MagicMock(
                    stop_reason="stop",
                    content=[MagicMock(text="Based on the search results, MCP is...")],
                ),
            ]
        )
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
        "lesson_count": 2,
    }


# Skip tests that require real ChromaDB if running in CI without docs
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require real dependencies)"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests (isolated, mocked)")
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may require network or heavy computation)"
    )
