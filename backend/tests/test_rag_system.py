"""
Integration tests for RAGSystem in rag_system.py.

Tests the end-to-end query flow:
- Query processing through the full system
- Session management
- Source tracking
- Exception propagation
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from rag_system import RAGSystem


@pytest.mark.unit
class TestRAGSystemInit:
    """Test RAGSystem initialization."""

    def test_init_creates_components(self, mock_config):
        """Test that initialization creates all required components."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            system = RAGSystem(mock_config)

            assert system.document_processor is not None
            assert system.vector_store is not None
            assert system.ai_generator is not None
            assert system.session_manager is not None
            assert system.tool_manager is not None

    def test_init_registers_tools(self, mock_config):
        """Test that tools are registered on initialization."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            system = RAGSystem(mock_config)

            # Should have at least 2 tools registered (search and outline)
            assert len(system.tool_manager.tools) >= 2
            assert 'search_course_content' in system.tool_manager.tools
            assert 'get_course_outline' in system.tool_manager.tools


@pytest.mark.unit
class TestRAGSystemQuery:
    """Test RAGSystem.query method."""

    def test_query_returns_response_and_sources(self, mock_config, mock_vector_store):
        """Test that query returns a tuple of (response, sources)."""
        with patch('rag_system.DocumentProcessor') as mock_dp, \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            # Mock the AI generator to return a response
            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(return_value="Test response")
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(mock_config)
            # Replace the AI with our mock
            system.ai_generator = mock_ai_instance

            response, sources = system.query("What is MCP?")

            assert response is not None
            assert isinstance(response, str)
            assert isinstance(sources, list)

    def test_query_without_session(self, mock_config, mock_vector_store):
        """Test query without providing a session_id."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(return_value="Response")
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai_instance

            response, sources = system.query("Test query")

            # Should still work without a session
            assert response is not None

    def test_query_with_session(self, mock_config, mock_vector_store):
        """Test query with an existing session_id."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(return_value="Response")
            mock_ai.return_value = mock_ai_instance

            mock_sm_instance = MagicMock()
            mock_sm_instance.get_conversation_history = Mock(return_value="Previous: Hi")
            mock_sm.return_value = mock_sm_instance

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai_instance
            system.session_manager = mock_sm_instance

            response, sources = system.query("Test query", session_id="session_1")

            # Should use conversation history
            mock_ai_instance.generate_response.assert_called_once()
            call_kwargs = mock_ai_instance.generate_response.call_args.kwargs
            assert 'conversation_history' in call_kwargs

    def test_query_updates_conversation_history(self, mock_config, mock_vector_store):
        """Test that query updates conversation history."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(return_value="Response")
            mock_ai.return_value = mock_ai_instance

            mock_sm_instance = MagicMock()
            mock_sm.return_value = mock_sm_instance

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai_instance
            system.session_manager = mock_sm_instance

            response, sources = system.query("Test query", session_id="session_1")

            # Should add exchange to session
            mock_sm_instance.add_exchange.assert_called_once_with(
                "session_1", "Test query", "Response"
            )

    def test_query_passes_tools_to_ai(self, mock_config, mock_vector_store):
        """Test that query passes tool definitions to AI generator."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(return_value="Response")
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai_instance

            system.query("Test query")

            # Check that tools were passed
            call_kwargs = mock_ai_instance.generate_response.call_args.kwargs
            assert 'tools' in call_kwargs
            assert 'tool_manager' in call_kwargs

    def test_query_returns_sources(self, mock_config, mock_vector_store):
        """Test that sources are returned from search tools."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(return_value="Response")
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai_instance

            # Simulate sources from search
            system.search_tool.last_sources = [
                {"text": "MCP Course", "url": "http://example.com"}
            ]

            response, sources = system.query("What is MCP?")

            assert isinstance(sources, list)
            # Sources should be retrieved and then reset
            assert len(system.search_tool.last_sources) == 0


@pytest.mark.unit
class TestRAGSystemQueryErrors:
    """Test error handling in RAGSystem.query."""

    def test_ai_generator_error_propagates(self, mock_config, mock_vector_store):
        """Test that AI generator errors propagate up."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(
                side_effect=Exception("API key invalid")
            )
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai_instance

            # Should raise the exception
            with pytest.raises(Exception, match="API key invalid"):
                system.query("Test query")

    def test_vector_store_error_propagates(self, mock_config):
        """Test that vector store errors propagate up."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            # Vector store that raises error
            mock_vs_instance = MagicMock()
            mock_vs_instance.search = Mock(side_effect=Exception("ChromaDB error"))
            mock_vs.return_value = mock_vs_instance

            mock_ai_instance = MagicMock()
            mock_ai_instance.generate_response = Mock(return_value="Response")
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(mock_config)
            system.vector_store = mock_vs_instance
            system.ai_generator = mock_ai_instance

            # The error propagates from vector store through tool to AI
            # But if AI doesn't use tool, it won't fail...
            # Let's test the tool directly
            system.search_tool.store = mock_vs_instance

            with pytest.raises(Exception, match="ChromaDB error"):
                system.search_tool.execute(query="test")


@pytest.mark.unit
class TestRAGSystemAddCourse:
    """Test RAGSystem.add_course_document method."""

    def test_add_course_document_success(self, mock_config, mock_vector_store):
        """Test adding a course document successfully."""
        with patch('rag_system.DocumentProcessor') as mock_dp, \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            from models import Course

            mock_course = Course(
                title="Test Course",
                instructor="Test Instructor",
                course_link="http://example.com",
                lessons=[]
            )
            mock_chunks = []

            mock_dp_instance = MagicMock()
            mock_dp_instance.process_course_document = Mock(return_value=(mock_course, mock_chunks))
            mock_dp.return_value = mock_dp_instance

            system = RAGSystem(mock_config)
            system.document_processor = mock_dp_instance

            course, chunk_count = system.add_course_document("/path/to/course.txt")

            assert course is not None
            assert chunk_count == 0

    def test_add_course_document_error(self, mock_config, mock_vector_store):
        """Test error handling when adding a course document."""
        with patch('rag_system.DocumentProcessor') as mock_dp, \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            mock_dp_instance = MagicMock()
            mock_dp_instance.process_course_document = Mock(
                side_effect=Exception("Parse error")
            )
            mock_dp.return_value = mock_dp_instance

            system = RAGSystem(mock_config)
            system.document_processor = mock_dp_instance

            course, chunk_count = system.add_course_document("/path/to/course.txt")

            # Should return None on error
            assert course is None
            assert chunk_count == 0


@pytest.mark.unit
class TestRAGSystemGetAnalytics:
    """Test RAGSystem.get_course_analytics method."""

    def test_get_course_analytics(self, mock_config):
        """Test getting course analytics."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            mock_vs_instance = MagicMock()
            mock_vs_instance.get_course_count = Mock(return_value=5)
            mock_vs_instance.get_existing_course_titles = Mock(
                return_value=["Course 1", "Course 2"]
            )
            mock_vs.return_value = mock_vs_instance

            system = RAGSystem(mock_config)
            system.vector_store = mock_vs_instance

            analytics = system.get_course_analytics()

            assert analytics["total_courses"] == 5
            assert analytics["course_titles"] == ["Course 1", "Course 2"]


@pytest.mark.integration
class TestRAGSystemEndToEnd:
    """End-to-end integration tests with more realistic mocks."""

    def test_full_query_flow_with_tool_use(self, mock_config):
        """Test the full query flow when Claude uses a tool."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm:

            # Mock vector store
            mock_vs_instance = MagicMock()
            mock_vs.return_value = mock_vs_instance

            # Mock AI generator with tool use flow
            mock_ai_instance = MagicMock()

            # First call returns tool use, second returns final response
            tool_use_response = MagicMock()
            tool_use_response.stop_reason = "tool_use"
            mock_tool = MagicMock()
            mock_tool.type = "tool_use"
            mock_tool.id = "toolu_123"
            mock_tool.name = "search_course_content"
            mock_tool.input = {"query": "MCP"}
            tool_use_response.content = [mock_tool]

            final_response = MagicMock()
            final_response.stop_reason = "stop"
            mock_content = MagicMock()
            mock_content.text = "MCP is a protocol..."
            final_response.content = [mock_content]

            call_count = [0]
            def side_effect_fn(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return tool_use_response
                return final_response

            mock_ai_instance.messages.create = Mock(side_effect=side_effect_fn)
            mock_ai_instance.generate_response = Mock(
                side_effect=lambda *args, **kwargs: mock_ai_instance._handle_tool_execution(
                    tool_use_response,
                    {
                        "messages": [{"role": "user", "content": kwargs.get("query", "")}],
                        "system": "System prompt"
                    },
                    MagicMock()
                ) if kwargs.get("tool_manager") else "Direct answer"
            )

            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai_instance
            system.vector_store = mock_vs_instance

            # Execute query
            response, sources = system.query("What is MCP?")

            # Should complete successfully
            assert response is not None
            assert isinstance(sources, list)
