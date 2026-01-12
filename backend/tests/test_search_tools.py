"""
Unit tests for CourseSearchTool in search_tools.py.

Tests the execute method with various scenarios:
- Valid queries returning results
- Course name filtering with semantic matching
- Lesson number filtering
- Empty results
- Error handling
- Output formatting
"""

import pytest
from unittest.mock import Mock, patch
from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


@pytest.mark.unit
class TestCourseSearchToolExecute:
    """Test the execute method of CourseSearchTool."""

    def test_valid_query_returns_formatted_results(self, course_search_tool):
        """Test that a valid query returns properly formatted results."""
        result = course_search_tool.execute(query="What is MCP?")

        assert result is not None
        assert isinstance(result, str)
        assert "MCP" in result
        # Check that the result contains formatted content
        assert "Test content" in result

    def test_query_with_course_name_uses_semantic_matching(
        self, course_search_tool, mock_vector_store
    ):
        """Test that course_name parameter is passed to search method."""
        # The search method should be called with course_name parameter
        mock_vector_store.search.reset_mock()

        course_search_tool.execute(query="MCP", course_name="MCP")

        # Verify search was called with course_name
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["course_name"] == "MCP"

    def test_query_with_lesson_number_filter(self, course_search_tool, mock_vector_store):
        """Test that lesson_number parameter is passed to search."""
        course_search_tool.execute(query="servers", lesson_number=2)

        # Verify search was called with lesson_number
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["lesson_number"] == 2

    def test_empty_results_returns_proper_message(self, mock_vector_store):
        """Test that empty search results return appropriate message."""
        # Configure mock to return empty results
        mock_vector_store.search = Mock(
            return_value=SearchResults(documents=[], metadata=[], distances=[], error=None)
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_empty_results_with_course_filter(self, mock_vector_store):
        """Test empty results message includes course name when filtered."""
        mock_vector_store.search = Mock(
            return_value=SearchResults(documents=[], metadata=[], distances=[], error=None)
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="topic", course_name="Some Course")

        assert "No relevant content found" in result
        assert "Some Course" in result

    def test_empty_results_with_lesson_filter(self, mock_vector_store):
        """Test empty results message includes lesson number when filtered."""
        mock_vector_store.search = Mock(
            return_value=SearchResults(documents=[], metadata=[], distances=[], error=None)
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="topic", lesson_number=5)

        assert "No relevant content found" in result
        assert "lesson 5" in result

    def test_vector_store_error_is_propagated(self, mock_vector_store):
        """Test that vector store errors are returned in the result."""
        error_msg = "Search error: ChromaDB connection failed"
        mock_vector_store.search = Mock(
            return_value=SearchResults(documents=[], metadata=[], distances=[], error=error_msg)
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert error_msg in result

    def test_course_not_found_error(self, mock_vector_store):
        """Test that non-existent course returns appropriate error."""
        # Configure _resolve_course_name to return None (not found)
        mock_vector_store._resolve_course_name.return_value = None
        # Make search return empty results
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="NonExistent Course")

        # When _resolve_course_name returns None, search returns empty SearchResults
        # which formats as "No relevant content found"
        assert "No relevant content found" in result


@pytest.mark.unit
class TestCourseSearchToolFormatResults:
    """Test the _format_results method of CourseSearchTool."""

    def test_format_results_structure(self, mock_vector_store, sample_search_results):
        """Test that _format_results produces correct structure."""
        tool = CourseSearchTool(mock_vector_store)
        result = tool._format_results(sample_search_results)

        # Should contain course title header
        assert "[" in result
        assert "]" in result
        # Should contain document content
        assert "MCP is a protocol" in result or "provides context" in result

    def test_format_results_updates_last_sources(self, mock_vector_store, sample_search_results):
        """Test that _format_results populates last_sources."""
        tool = CourseSearchTool(mock_vector_store)
        assert len(tool.last_sources) == 0

        tool._format_results(sample_search_results)

        # last_sources should be populated
        assert len(tool.last_sources) > 0
        # Each source should have 'text' and 'url' keys
        for source in tool.last_sources:
            assert "text" in source
            assert "url" in source

    def test_format_results_with_lesson_metadata(self, mock_vector_store):
        """Test formatting results with lesson number in metadata."""
        results_with_lesson = SearchResults(
            documents=["Lesson content about servers"],
            metadata=[{"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 2}],
            distances=[0.1],
            error=None,
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool._format_results(results_with_lesson)

        # Should show lesson number in the header
        assert "Lesson 2" in result


@pytest.mark.unit
class TestCourseOutlineTool:
    """Test the CourseOutlineTool class."""

    def test_get_outline_with_course_name(self, course_outline_tool, mock_vector_store):
        """Test getting outline for a specific course."""
        result = course_outline_tool.execute(course_name="MCP")

        assert result is not None
        assert isinstance(result, str)
        # Should contain course information
        assert "Course:" in result
        assert "Link:" in result
        assert "Lessons:" in result

    def test_get_outline_all_courses(self, course_outline_tool):
        """Test getting list of all courses when no course name provided."""
        result = course_outline_tool.execute()

        assert result is not None
        assert isinstance(result, str)
        assert "Available courses:" in result

    def test_get_outline_nonexistent_course(self, course_outline_tool, mock_vector_store):
        """Test outline request for non-existent course."""
        mock_vector_store._resolve_course_name = Mock(return_value=None)

        result = course_outline_tool.execute(course_name="NonExistent")

        assert "No course found matching" in result

    def test_outline_uses_semantic_matching(self, course_outline_tool, mock_vector_store):
        """Test that outline tool uses semantic course name matching."""
        course_outline_tool.execute(course_name="MCP")

        # Verify _resolve_course_name was called
        mock_vector_store._resolve_course_name.assert_called_once_with("MCP")


@pytest.mark.unit
class TestToolManager:
    """Test the ToolManager class."""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] is tool

    def test_register_multiple_tools(self, mock_vector_store):
        """Test registering multiple tools."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, tool_manager):
        """Test getting tool definitions."""
        definitions = tool_manager.get_tool_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) == 2  # search and outline tools
        # Check each definition has required keys
        for definition in definitions:
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition

    def test_execute_tool_success(self, tool_manager):
        """Test executing a registered tool."""
        # Mock the tool's execute method
        for tool in tool_manager.tools.values():
            tool.execute = Mock(return_value="Test result")

        result = tool_manager.execute_tool("search_course_content", query="test")

        assert result == "Test result"

    def test_execute_tool_not_found(self, tool_manager):
        """Test executing a non-existent tool."""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources(self, mock_vector_store):
        """Test getting sources from last search."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)

        # Simulate a search that populates last_sources
        search_tool.last_sources = [{"text": "Test Course", "url": "http://example.com"}]

        manager.register_tool(search_tool)
        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Test Course"

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources from all tools."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [{"text": "Test", "url": "http://example.com"}]

        manager.register_tool(search_tool)
        manager.reset_sources()

        assert len(search_tool.last_sources) == 0

    def test_tool_without_last_sources_attribute(self, mock_vector_store):
        """Test get_last_sources when tool doesn't have last_sources."""
        manager = ToolManager()

        # Create a mock tool without last_sources
        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(
            return_value={
                "name": "mock_tool",
                "description": "Mock tool",
                "input_schema": {"type": "object"},
            }
        )
        del mock_tool.last_sources  # Remove last_sources attribute

        manager.register_tool(mock_tool)
        sources = manager.get_last_sources()

        # Should return empty list without error
        assert sources == []


@pytest.mark.unit
class TestToolDefinitions:
    """Test tool definition structures."""

    def test_course_search_tool_definition(self, course_search_tool):
        """Test that CourseSearchTool has valid definition."""
        definition = course_search_tool.get_tool_definition()

        assert "name" in definition
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "properties" in definition["input_schema"]
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["properties"]["query"]["type"] == "string"

    def test_course_outline_tool_definition(self, course_outline_tool):
        """Test that CourseOutlineTool has valid definition."""
        definition = course_outline_tool.get_tool_definition()

        assert "name" in definition
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert "properties" in definition["input_schema"]
        # course_name should be optional
        assert "course_name" in definition["input_schema"]["properties"]


@pytest.mark.unit
class TestToolErrorHandling:
    """Test error handling in tools."""

    def test_search_tool_handles_chromadb_exception(self, mock_vector_store):
        """Test that search tool does NOT catch ChromaDB exceptions - they propagate."""
        # Make search raise an exception
        mock_vector_store.search = Mock(side_effect=Exception("ChromaDB crash"))

        tool = CourseSearchTool(mock_vector_store)
        # The exception will propagate up - this is a potential cause of "query failed"
        with pytest.raises(Exception, match="ChromaDB crash"):
            tool.execute(query="test")

    def test_outline_tool_handles_json_parse_error(self, mock_vector_store):
        """Test that outline tool handles invalid JSON in lessons."""
        mock_vector_store._resolve_course_name = Mock(return_value="Test Course")
        mock_catalog = Mock()
        mock_catalog.get = Mock(
            return_value={
                "metadatas": [
                    {
                        "title": "Test Course",
                        "course_link": "http://example.com",
                        "lessons_json": "invalid json{{{",
                        "lesson_count": 2,
                    }
                ]
            }
        )
        mock_vector_store.course_catalog = mock_catalog

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Test Course")

        # Should handle JSON parse error
        assert "Error" in result or "error" in result
