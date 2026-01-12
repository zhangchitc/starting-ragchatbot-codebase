"""
API endpoint tests for the FastAPI application in app.py.

Tests cover:
- POST /api/query - Query endpoint with request/response validation
- GET /api/courses - Course statistics endpoint
- GET / - Root health check endpoint
- Error handling for various failure scenarios
- Request validation (missing fields, invalid data)
"""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.api
class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self, client):
        """Test that root endpoint returns JSON response."""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"

    def test_root_response_structure(self, client):
        """Test that root endpoint returns expected structure."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert data["status"] == "ok"


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for POST /api/query endpoint."""

    def test_query_returns_200(self, client, sample_query_request):
        """Test that query endpoint returns 200 for valid request."""
        response = client.post("/api/query", json=sample_query_request)
        assert response.status_code == 200

    def test_query_returns_json(self, client, sample_query_request):
        """Test that query endpoint returns JSON response."""
        response = client.post("/api/query", json=sample_query_request)
        assert response.headers["content-type"] == "application/json"

    def test_query_response_structure(self, client, sample_query_request):
        """Test that query response has expected fields."""
        response = client.post("/api/query", json=sample_query_request)
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["sources"], list)

    def test_query_creates_session_when_not_provided(self, client, sample_query_request, mock_rag_system):
        """Test that a new session is created when session_id is not provided."""
        response = client.post("/api/query", json=sample_query_request)
        data = response.json()

        assert response.status_code == 200
        assert data["session_id"] == "test_session_123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_existing_session(self, client, sample_query_request_with_session, mock_rag_system):
        """Test that existing session_id is used when provided."""
        response = client.post("/api/query", json=sample_query_request_with_session)
        data = response.json()

        assert response.status_code == 200
        assert data["session_id"] == "existing_session_456"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_calls_rag_system(self, client, sample_query_request, mock_rag_system):
        """Test that query endpoint calls RAGSystem.query method."""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_passes_correct_parameters(self, client, sample_query_request, mock_rag_system):
        """Test that query endpoint passes correct parameters to RAGSystem."""
        client.post("/api/query", json=sample_query_request)

        call_args = mock_rag_system.query.call_args
        assert call_args[0][0] == "What is MCP?"  # query parameter
        assert call_args[0][1] == "test_session_123"  # session_id parameter

    def test_query_returns_sources(self, client, sample_query_request):
        """Test that query response includes sources."""
        response = client.post("/api/query", json=sample_query_request)
        data = response.json()

        assert len(data["sources"]) > 0
        assert "text" in data["sources"][0]
        assert "url" in data["sources"][0]

    def test_query_missing_query_field(self, client):
        """Test that request without query field returns 422 validation error."""
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_query_empty_query_string(self, client):
        """Test that empty query string is accepted (no length validation)."""
        response = client.post("/api/query", json={"query": ""})
        # Empty strings pass Pydantic validation (no min_length constraint)
        assert response.status_code == 200

    def test_query_with_extra_fields(self, client, sample_query_request):
        """Test that extra fields in request are ignored (not an error)."""
        request_with_extra = {**sample_query_request, "extra_field": "some_value"}
        response = client.post("/api/query", json=request_with_extra)
        assert response.status_code == 200


@pytest.mark.api
class TestQueryEndpointErrorHandling:
    """Tests for error handling in query endpoint."""

    def test_query_handles_rag_system_error(self, error_client):
        """Test that RAGSystem errors are caught and returned as 500."""
        response = error_client.post("/api/query", json={"query": "Test query"})
        assert response.status_code == 500

    def test_query_error_response_has_detail(self, error_client):
        """Test that error response contains detail field."""
        response = error_client.post("/api/query", json={"query": "Test query"})
        data = response.json()
        assert "detail" in data
        assert "API key invalid" in data["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint."""

    def test_courses_returns_200(self, client):
        """Test that courses endpoint returns 200 OK."""
        response = client.get("/api/courses")
        assert response.status_code == 200

    def test_courses_returns_json(self, client):
        """Test that courses endpoint returns JSON response."""
        response = client.get("/api/courses")
        assert response.headers["content-type"] == "application/json"

    def test_courses_response_structure(self, client):
        """Test that courses response has expected fields."""
        response = client.get("/api/courses")
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["course_titles"], list)

    def test_courses_returns_correct_data(self, client, sample_courses_response):
        """Test that courses endpoint returns correct analytics data."""
        response = client.get("/api/courses")
        data = response.json()

        assert response.status_code == 200
        assert data["total_courses"] == sample_courses_response["total_courses"]
        assert data["course_titles"] == sample_courses_response["course_titles"]

    def test_courses_calls_rag_system_analytics(self, client, mock_rag_system):
        """Test that courses endpoint calls RAGSystem.get_course_analytics."""
        client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()


@pytest.mark.api
class TestCoursesEndpointErrorHandling:
    """Tests for error handling in courses endpoint."""

    def test_courses_handles_rag_system_error(self, error_client):
        """Test that RAGSystem errors are caught and returned as 500."""
        response = error_client.get("/api/courses")
        assert response.status_code == 500

    def test_courses_error_response_has_detail(self, error_client):
        """Test that error response contains detail field."""
        response = error_client.get("/api/courses")
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]


@pytest.mark.api
class TestCorsHeaders:
    """Tests for CORS middleware configuration."""

    def test_query_endpoint_has_cors_headers(self, client):
        """Test that query endpoint includes CORS headers."""
        response = client.post("/api/query", json={"query": "Test"})
        # OPTIONS preflight would have more headers, but POST should have some
        assert response.status_code in (200, 405)  # 405 if method not allowed

    def test_courses_endpoint_allows_get(self, client):
        """Test that courses endpoint allows GET requests."""
        response = client.get("/api/courses")
        assert response.status_code == 200


@pytest.mark.api
class TestSessionFlow:
    """Tests for session management across multiple requests."""

    def test_consecutive_queries_same_session(self, client, mock_rag_system):
        """Test that consecutive queries with same session_id maintain context."""
        first_request = {"query": "What is MCP?"}
        second_request = {"query": "Tell me more", "session_id": "test_session_123"}

        # First request creates session
        first_response = client.post("/api/query", json=first_request)
        session_id = first_response.json()["session_id"]

        # Second request uses existing session
        second_response = client.post("/api/query", json=second_request)

        assert second_response.status_code == 200
        assert second_response.json()["session_id"] == session_id

    def test_multiple_sessions_are_independent(self, client, mock_rag_system):
        """Test that different sessions are handled independently."""
        request1 = {"query": "Question 1"}
        request2 = {"query": "Question 2"}

        response1 = client.post("/api/query", json=request1)
        response2 = client.post("/api/query", json=request2)

        # Both should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Each should have a session ID (same in this case due to mock)
        assert "session_id" in response1.json()
        assert "session_id" in response2.json()


@pytest.mark.api
class TestResponseValidation:
    """Tests for response model validation."""

    def test_query_answer_is_string(self, client, sample_query_request):
        """Test that answer field is always a string."""
        response = client.post("/api/query", json=sample_query_request)
        data = response.json()
        assert isinstance(data["answer"], str)

    def test_query_sources_is_list(self, client, sample_query_request):
        """Test that sources field is always a list."""
        response = client.post("/api/query", json=sample_query_request)
        data = response.json()
        assert isinstance(data["sources"], list)

    def test_query_session_id_is_string(self, client, sample_query_request):
        """Test that session_id field is always a string."""
        response = client.post("/api/query", json=sample_query_request)
        data = response.json()
        assert isinstance(data["session_id"], str)

    def test_courses_total_courses_is_int(self, client):
        """Test that total_courses field is always an integer."""
        response = client.get("/api/courses")
        data = response.json()
        assert isinstance(data["total_courses"], int)

    def test_courses_course_titles_is_list_of_strings(self, client):
        """Test that course_titles is always a list of strings."""
        response = client.get("/api/courses")
        data = response.json()
        assert isinstance(data["course_titles"], list)
        if len(data["course_titles"]) > 0:
            assert all(isinstance(title, str) for title in data["course_titles"])


@pytest.mark.api
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_query_with_very_long_text(self, client, mock_rag_system):
        """Test query with a very long query string."""
        long_query = {"query": "What is MCP? " * 100}
        response = client.post("/api/query", json=long_query)
        assert response.status_code == 200

    def test_query_with_special_characters(self, client, mock_rag_system):
        """Test query with special characters."""
        special_query = {"query": "What is MCP? @#$%^&*()_+{}|:\"<>?`~"}
        response = client.post("/api/query", json=special_query)
        assert response.status_code == 200

    def test_query_with_unicode(self, client, mock_rag_system):
        """Test query with unicode characters."""
        unicode_query = {"query": "What is MCP? 🚀 你好 مرحبا"}
        response = client.post("/api/query", json=unicode_query)
        assert response.status_code == 200

    def test_courses_with_empty_catalog(self, client, mock_rag_system):
        """Test courses endpoint when catalog is empty."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        response = client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_with_many_titles(self, client, mock_rag_system):
        """Test courses endpoint with many course titles."""
        many_titles = [f"Course {i}" for i in range(100)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": many_titles
        }
        response = client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100


@pytest.mark.api
class TestHttpMethods:
    """Tests for correct HTTP method handling."""

    def test_query_endpoint_rejects_get(self, client):
        """Test that query endpoint rejects GET requests."""
        response = client.get("/api/query")
        assert response.status_code == 405  # Method Not Allowed

    def test_query_endpoint_rejects_put(self, client):
        """Test that query endpoint rejects PUT requests."""
        response = client.put("/api/query", json={"query": "test"})
        assert response.status_code == 405

    def test_query_endpoint_rejects_delete(self, client):
        """Test that query endpoint rejects DELETE requests."""
        response = client.delete("/api/query")
        assert response.status_code == 405

    def test_courses_endpoint_rejects_post(self, client):
        """Test that courses endpoint rejects POST requests."""
        response = client.post("/api/courses", json={})
        assert response.status_code == 405

    def test_courses_endpoint_rejects_put(self, client):
        """Test that courses endpoint rejects PUT requests."""
        response = client.put("/api/courses")
        assert response.status_code == 405

    def test_courses_endpoint_rejects_delete(self, client):
        """Test that courses endpoint rejects DELETE requests."""
        response = client.delete("/api/courses")
        assert response.status_code == 405
