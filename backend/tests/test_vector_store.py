"""
Unit tests for VectorStore in vector_store.py.

Tests the search functionality and metadata handling:
- Search returns valid results
- Semantic course name matching
- Error handling
- Metadata retrieval
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from vector_store import VectorStore, SearchResults


@pytest.mark.unit
class TestSearchResults:
    """Test SearchResults dataclass."""

    def test_create_empty_results(self):
        """Test creating empty SearchResults."""
        results = SearchResults.empty("Test error")
        assert results.error == "Test error"
        assert results.is_empty() is True

    def test_is_empty_true(self):
        """Test is_empty returns True when no documents."""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False when documents exist."""
        results = SearchResults(
            documents=["Test doc"],
            metadata=[{"test": "value"}],
            distances=[0.1]
        )
        assert results.is_empty() is False


@pytest.mark.unit
class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_init_with_temp_path(self, mock_config):
        """Test initialization with a temporary path."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )

            assert store.max_results == mock_config.MAX_RESULTS

    def test_init_creates_collections(self, mock_config):
        """Test that initialization creates the required collections."""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_chroma_client = MagicMock()
            mock_client.return_value = mock_chroma_client

            mock_collection = MagicMock()
            mock_chroma_client.get_or_create_collection = Mock(return_value=mock_collection)

            with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                store = VectorStore(
                    chroma_path=mock_config.CHROMA_PATH,
                    embedding_model=mock_config.EMBEDDING_MODEL
                )

                # Should create two collections
                assert mock_chroma_client.get_or_create_collection.call_count == 2


@pytest.mark.unit
class TestVectorStoreSearch:
    """Test VectorStore.search method."""

    def test_search_with_query_only(self, mock_config):
        """Test search with just a query parameter."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            # Create a real SearchResults for the mock to return
            mock_results = {
                'documents': [['Test content']],
                'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
                'distances': [[0.1]]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_content = MagicMock()
            store.course_content.query = Mock(return_value=mock_results)

            results = store.search("test query")

            assert results.is_empty() is False
            assert len(results.documents) == 1

    def test_search_with_course_filter(self, mock_config):
        """Test search with course_name filter."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            mock_results = {
                'documents': [['Test content']],
                'metadatas': [[{'course_title': 'MCP Course', 'lesson_number': 1}]],
                'distances': [[0.1]]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_content = MagicMock()
            store.course_content.query = Mock(return_value=mock_results)

            # Mock _resolve_course_name
            store._resolve_course_name = Mock(return_value="MCP Course")

            results = store.search("test query", course_name="MCP")

            # Check that the filter was applied
            call_kwargs = store.course_content.query.call_args.kwargs
            assert 'where' in call_kwargs
            assert call_kwargs['where']['course_title'] == "MCP Course"

    def test_search_with_lesson_filter(self, mock_config):
        """Test search with lesson_number filter."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            mock_results = {
                'documents': [['Test content']],
                'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 2}]],
                'distances': [[0.1]]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_content = MagicMock()
            store.course_content.query = Mock(return_value=mock_results)

            results = store.search("test query", lesson_number=2)

            # Check that the filter was applied
            call_kwargs = store.course_content.query.call_args.kwargs
            assert 'where' in call_kwargs
            assert call_kwargs['where']['lesson_number'] == 2

    def test_search_course_not_found(self, mock_config):
        """Test search when course is not found."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            # Mock _resolve_course_name to return None
            store._resolve_course_name = Mock(return_value=None)

            results = store.search("test query", course_name="NonExistent")

            assert results.is_empty()
            assert "No course found matching" in results.error

    def test_search_with_limit(self, mock_config):
        """Test search with custom limit."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            mock_results = {
                'documents': [['Test content']],
                'metadatas': [[{'test': 'value'}]],
                'distances': [[0.1]]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=5
            )
            store.course_content = MagicMock()
            store.course_content.query = Mock(return_value=mock_results)

            store.search("test query", limit=3)

            # Check that custom limit was used
            call_kwargs = store.course_content.query.call_args.kwargs
            assert call_kwargs['n_results'] == 3

    def test_search_error_propagation(self, mock_config):
        """Test that search errors are caught and returned in SearchResults."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_content = MagicMock()
            store.course_content.query = Mock(side_effect=Exception("ChromaDB error"))

            results = store.search("test query")

            assert results.error is not None
            assert "ChromaDB error" in results.error


@pytest.mark.unit
class TestVectorStoreResolveCourseName:
    """Test _resolve_course_name method."""

    def test_resolve_course_name_found(self, mock_config):
        """Test resolving a course name that exists."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            mock_results = {
                'documents': [['some text']],
                'metadatas': [[{'title': 'MCP: Build Rich-Context AI Apps'}]],
                'distances': [[0.1]]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_catalog = MagicMock()
            store.course_catalog.query = Mock(return_value=mock_results)

            result = store._resolve_course_name("MCP")

            assert result == "MCP: Build Rich-Context AI Apps"

    def test_resolve_course_name_not_found(self, mock_config):
        """Test resolving a course name that doesn't exist."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            # Empty results
            mock_results = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_catalog = MagicMock()
            store.course_catalog.query = Mock(return_value=mock_results)

            result = store._resolve_course_name("NonExistent")

            assert result is None

    def test_resolve_course_name_error_handling(self, mock_config):
        """Test error handling in resolve course name."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_catalog = MagicMock()
            store.course_catalog.query = Mock(side_effect=Exception("Query error"))

            result = store._resolve_course_name("Test")

            assert result is None


@pytest.mark.unit
class TestVectorStoreBuildFilter:
    """Test _build_filter method."""

    def test_build_filter_no_filters(self, mock_config):
        """Test building filter with no constraints."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )

            result = store._build_filter(None, None)
            assert result is None

    def test_build_filter_course_only(self, mock_config):
        """Test building filter with course only."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )

            result = store._build_filter("Test Course", None)
            assert result == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, mock_config):
        """Test building filter with lesson only."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )

            result = store._build_filter(None, 5)
            assert result == {"lesson_number": 5}

    def test_build_filter_both(self, mock_config):
        """Test building filter with both course and lesson."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )

            result = store._build_filter("Test Course", 5)
            assert result == {"$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 5}
            ]}


@pytest.mark.unit
class TestVectorStoreMetadata:
    """Test metadata retrieval methods."""

    def test_get_course_link(self, mock_config):
        """Test getting course link."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            mock_results = {
                'metadatas': [{'course_link': 'https://example.com/course'}]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_catalog = MagicMock()
            store.course_catalog.get = Mock(return_value=mock_results)

            result = store.get_course_link("Test Course")
            assert result == 'https://example.com/course'

    def test_get_lesson_link(self, mock_config):
        """Test getting lesson link."""
        import json
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            lessons = [
                {"lesson_number": 0, "lesson_link": "link0"},
                {"lesson_number": 1, "lesson_link": "link1"},
            ]

            mock_results = {
                'metadatas': [{'lessons_json': json.dumps(lessons)}]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_catalog = MagicMock()
            store.course_catalog.get = Mock(return_value=mock_results)

            result = store.get_lesson_link("Test Course", 1)
            assert result == 'link1'

    def test_get_all_courses_metadata(self, mock_config):
        """Test getting all courses metadata."""
        import json
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            lessons = [{"lesson_number": 0, "lesson_title": "Intro"}]

            mock_results = {
                'metadatas': [
                    {
                        'title': 'Test Course',
                        'instructor': 'Test Instructor',
                        'course_link': 'https://example.com',
                        'lessons_json': json.dumps(lessons),
                        'lesson_count': 1
                    }
                ]
            }

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_catalog = MagicMock()
            store.course_catalog.get = Mock(return_value=mock_results)

            result = store.get_all_courses_metadata()

            assert len(result) == 1
            assert result[0]['title'] == 'Test Course'
            assert 'lessons' in result[0]
            assert result[0]['lessons'][0]['lesson_title'] == 'Intro'


@pytest.mark.unit
class TestVectorStoreErrors:
    """Test error handling in VectorStore."""

    def test_search_exception_returns_error_result(self, mock_config):
        """Test that search exceptions are caught and returned as error."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_content = MagicMock()
            store.course_content.query = Mock(
                side_effect=Exception("Connection failed")
            )

            results = store.search("test query")

            assert results.error is not None
            assert "Connection failed" in results.error

    def test_get_course_link_handles_errors(self, mock_config):
        """Test error handling in get_course_link."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL
            )
            store.course_catalog = MagicMock()
            store.course_catalog.get = Mock(side_effect=Exception("DB error"))

            result = store.get_course_link("Test Course")
            assert result is None
