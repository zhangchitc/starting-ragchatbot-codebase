"""
Unit tests for AIGenerator in ai_generator.py.

Tests the generate_response method and sequential tool calling behavior:
- Direct responses (no tool use)
- Single tool use responses
- Sequential tool calling (up to 2 rounds)
- Error handling for API failures
- System prompt handling
- Termination conditions (max rounds, no tool use, errors)
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from ai_generator import AIGenerator


@pytest.mark.unit
class TestAIGeneratorInit:
    """Test AIGenerator initialization."""

    def test_init_with_valid_params(self, mock_config):
        """Test initialization with valid parameters."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL,
                max_tool_rounds=2
            )

            assert generator.model == mock_config.ANTHROPIC_MODEL
            assert generator.client is not None
            assert generator.max_tool_rounds == 2

    def test_init_with_custom_max_rounds(self, mock_config):
        """Test initialization with custom max_tool_rounds."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL,
                max_tool_rounds=5
            )

            assert generator.max_tool_rounds == 5

    def test_base_params_setup(self, mock_config):
        """Test that base parameters are set up correctly."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )

            assert generator.base_params['model'] == mock_config.ANTHROPIC_MODEL
            assert generator.base_params['temperature'] == 0
            assert generator.base_params['max_tokens'] == 800


@pytest.mark.unit
class TestGenerateResponseNoTools:
    """Test generate_response without tool use (direct answer)."""

    def test_direct_answer_response(self, ai_generator_with_mock):
        """Test that a direct answer is returned when no tool use is needed."""
        response = ai_generator_with_mock.generate_response("What is Python?")

        assert response is not None
        assert isinstance(response, str)
        assert "test response" in response.lower()

    def test_system_prompt_passed_without_history(self, ai_generator_with_mock, mock_config):
        """Test that system prompt is passed correctly without conversation history."""
        ai_generator_with_mock.generate_response("Test query")

        # Check that the API was called
        ai_generator_with_mock.client.messages.create.assert_called_once()
        call_args = ai_generator_with_mock.client.messages.create.call_args
        assert 'system' in call_args.kwargs
        assert AIGenerator.SYSTEM_PROMPT in call_args.kwargs['system']

    def test_system_prompt_passed_with_history(self, ai_generator_with_mock):
        """Test that system prompt includes conversation history."""
        history = "User: Hi\nAI: Hello"
        ai_generator_with_mock.generate_response("Test query", conversation_history=history)

        call_args = ai_generator_with_mock.client.messages.create.call_args
        assert 'system' in call_args.kwargs
        assert AIGenerator.SYSTEM_PROMPT in call_args.kwargs['system']
        assert history in call_args.kwargs['system']

    def test_no_tools_without_tools_param(self, ai_generator_with_mock):
        """Test that tools are not added when not provided."""
        ai_generator_with_mock.generate_response("What is Python?")

        call_args = ai_generator_with_mock.client.messages.create.call_args
        # Tools should not be in the call
        assert 'tools' not in call_args.kwargs or call_args.kwargs.get('tools') is None


@pytest.mark.unit
class TestSingleToolUse:
    """Test single tool use (one round)."""

    def test_single_tool_use_flow(self, mock_config, tool_manager):
        """Test a single tool use followed by final answer."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()

            # First call: tool use
            tool_use_response = MagicMock()
            tool_use_response.stop_reason = "tool_use"
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.id = "toolu_123"
            mock_tool_use.name = "search_course_content"
            mock_tool_use.input = {"query": "MCP"}
            tool_use_response.content = [mock_tool_use]

            # Second call: final answer (no tool use)
            final_response = MagicMock()
            final_response.stop_reason = "stop"
            mock_content = MagicMock()
            mock_content.text = "MCP is a protocol for AI tools."
            final_response.content = [mock_content]

            mock_client.messages.create = Mock(side_effect=[tool_use_response, final_response])
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL,
                max_tool_rounds=2
            )
            generator.client = mock_client

            tool_manager.execute_tool = Mock(return_value="Search results")

            result = generator.generate_response(
                "What is MCP?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Should have called API twice (tool use + final)
            assert mock_client.messages.create.call_count == 2
            # Tool was executed once
            tool_manager.execute_tool.assert_called_once()
            assert "protocol" in result.lower()

    def test_tools_added_to_api_call(self, ai_generator_with_mock, tool_manager):
        """Test that tools are added to the API call when provided."""
        ai_generator_with_mock.generate_response(
            "What is MCP?",
            tools=tool_manager.get_tool_definitions()
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        assert 'tools' in call_args.kwargs
        assert 'tool_choice' in call_args.kwargs
        assert call_args.kwargs['tool_choice'] == {"type": "auto"}


@pytest.mark.unit
class TestSequentialToolCalling:
    """Test sequential tool calling (multiple rounds)."""

    def test_two_round_sequential_tool_calls(self, mock_config, tool_manager):
        """Test two sequential tool calls (outline then search)."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()

            # Round 1: Claude calls get_course_outline
            outline_response = MagicMock()
            outline_response.stop_reason = "tool_use"
            mock_outline_tool = MagicMock()
            mock_outline_tool.type = "tool_use"
            mock_outline_tool.id = "toolu_111"
            mock_outline_tool.name = "get_course_outline"
            mock_outline_tool.input = {"course_name": "MCP"}
            outline_response.content = [mock_outline_tool]

            # Round 2: Claude calls search_course_content
            search_response = MagicMock()
            search_response.stop_reason = "tool_use"
            mock_search_tool = MagicMock()
            mock_search_tool.type = "tool_use"
            mock_search_tool.id = "toolu_222"
            mock_search_tool.name = "search_course_content"
            mock_search_tool.input = {"query": "Building MCP Servers"}
            search_response.content = [mock_search_tool]

            # Final: Claude synthesizes answer
            final_response = MagicMock()
            final_response.stop_reason = "stop"
            mock_final_content = MagicMock()
            mock_final_content.text = "Based on the outline and search, here's what I found..."
            final_response.content = [mock_final_content]

            # Set up call sequence
            mock_client.messages.create = Mock(side_effect=[
                outline_response,
                search_response,
                final_response
            ])
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL,
                max_tool_rounds=2
            )
            generator.client = mock_client

            # Mock tool manager to return different results per call
            tool_results = ["Lesson 4: Building MCP Servers", "Search results about servers"]
            tool_manager.execute_tool = Mock(side_effect=tool_results)

            result = generator.generate_response(
                "What does lesson 4 of MCP cover, and are there similar courses?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Should have called API 3 times (2 tool rounds + 1 final)
            assert mock_client.messages.create.call_count == 3
            # Both tools were executed
            assert tool_manager.execute_tool.call_count == 2
            # Result contains the synthesized answer
            assert "found" in result.lower()

    def test_max_rounds_termination(self, mock_config, tool_manager):
        """Test that tool calling stops after max_rounds is reached."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()

            # All responses request tool use
            def create_tool_response():
                resp = MagicMock()
                resp.stop_reason = "tool_use"
                mock_tool = MagicMock()
                mock_tool.type = "tool_use"
                mock_tool.id = "toolu_xxx"
                mock_tool.name = "search_course_content"
                mock_tool.input = {"query": "test"}
                resp.content = [mock_tool]
                return resp

            tool_responses = [create_tool_response() for _ in range(3)]
            # Final response for forced synthesis
            final_response = MagicMock()
            final_response.stop_reason = "stop"
            mock_final_content = MagicMock()
            mock_final_content.text = "I've gathered information from multiple searches."
            final_response.content = [mock_final_content]

            # 3 tool calls + 1 final synthesis
            mock_client.messages.create = Mock(side_effect=tool_responses + [final_response])
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL,
                max_tool_rounds=2  # Limit to 2 rounds
            )
            generator.client = mock_client

            tool_manager.execute_tool = Mock(return_value="Results")

            result = generator.generate_response(
                "Keep searching for more information",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Should stop after 2 rounds + 1 final = 3 API calls
            assert mock_client.messages.create.call_count == 3

    def test_early_termination_no_tool_use(self, mock_config, tool_manager):
        """Test that tool calling stops early when Claude doesn't request tools."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()

            # First response: direct answer (no tool use)
            direct_response = MagicMock()
            direct_response.stop_reason = "stop"
            mock_content = MagicMock()
            mock_content.text = "Python is a programming language."
            direct_response.content = [mock_content]

            mock_client.messages.create = Mock(return_value=direct_response)
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL,
                max_tool_rounds=2
            )
            generator.client = mock_client

            result = generator.generate_response(
                "What is Python?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Should only call API once (no tool use)
            assert mock_client.messages.create.call_count == 1
            # No tools executed
            tool_manager.execute_tool.assert_not_called()
            assert "programming language" in result.lower()


@pytest.mark.unit
class TestToolExecutionErrors:
    """Test error handling during tool execution."""

    def test_tool_execution_error_terminates_gracefully(self, mock_config, tool_manager):
        """Test that tool execution errors force final synthesis with error info."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()

            # Tool use response
            tool_use_response = MagicMock()
            tool_use_response.stop_reason = "tool_use"
            mock_tool = MagicMock()
            mock_tool.type = "tool_use"
            mock_tool.id = "toolu_123"
            mock_tool.name = "search_course_content"
            mock_tool.input = {"query": "test"}
            tool_use_response.content = [mock_tool]

            # Final synthesis response
            final_response = MagicMock()
            final_response.stop_reason = "stop"
            mock_content = MagicMock()
            mock_content.text = "I encountered an error while searching."
            final_response.content = [mock_content]

            mock_client.messages.create = Mock(side_effect=[tool_use_response, final_response])
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL,
                max_tool_rounds=2
            )
            generator.client = mock_client

            # Tool execution raises exception
            tool_manager.execute_tool = Mock(side_effect=Exception("Tool failed"))

            result = generator.generate_response(
                "Search for something",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Should call API twice (tool use + final synthesis)
            assert mock_client.messages.create.call_count == 2
            # Should return a response despite the error
            assert result is not None


@pytest.mark.unit
class TestExecuteToolsFromResponse:
    """Test the _execute_tools_from_response helper method."""

    def test_execute_single_tool(self, mock_config):
        """Test executing a single tool from a response."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )

            # Create mock response with tool use
            response = MagicMock()
            mock_tool = MagicMock()
            mock_tool.type = "tool_use"
            mock_tool.id = "toolu_123"
            mock_tool.name = "test_tool"
            mock_tool.input = {"arg": "value"}
            response.content = [mock_tool]

            tool_manager = Mock()
            tool_manager.execute_tool = Mock(return_value="Tool result")

            results, had_error = generator._execute_tools_from_response(response, tool_manager)

            assert len(results) == 1
            assert results[0]["type"] == "tool_result"
            assert results[0]["tool_use_id"] == "toolu_123"
            assert results[0]["content"] == "Tool result"
            assert had_error is False

    def test_execute_multiple_tools(self, mock_config):
        """Test executing multiple tools from a response."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )

            # Create mock response with 2 tools
            response = MagicMock()
            tool1 = MagicMock()
            tool1.type = "tool_use"
            tool1.id = "toolu_1"
            tool1.name = "tool1"
            tool1.input = {"query": "test1"}

            tool2 = MagicMock()
            tool2.type = "tool_use"
            tool2.id = "toolu_2"
            tool2.name = "tool2"
            tool2.input = {"query": "test2"}

            response.content = [tool1, tool2]

            tool_manager = Mock()
            tool_manager.execute_tool = Mock(return_value="Result")

            results, had_error = generator._execute_tools_from_response(response, tool_manager)

            assert len(results) == 2
            assert tool_manager.execute_tool.call_count == 2
            assert had_error is False

    def test_tool_execution_error_handling(self, mock_config):
        """Test that tool execution errors are caught and marked."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )

            response = MagicMock()
            mock_tool = MagicMock()
            mock_tool.type = "tool_use"
            mock_tool.id = "toolu_123"
            mock_tool.name = "failing_tool"
            mock_tool.input = {}
            response.content = [mock_tool]

            tool_manager = Mock()
            tool_manager.execute_tool = Mock(side_effect=Exception("Tool error"))

            results, had_error = generator._execute_tools_from_response(response, tool_manager)

            assert len(results) == 1
            assert "Error" in results[0]["content"]
            assert had_error is True


@pytest.mark.unit
class TestForceFinalSynthesis:
    """Test the _force_final_synthesis helper method."""

    def test_force_final_synthesis_calls_api(self, mock_config):
        """Test that _force_final_synthesis makes an API call without tools."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()

            final_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Here's my answer."
            final_response.content = [mock_content]

            mock_client.messages.create = Mock(return_value=final_response)
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )
            generator.client = mock_client

            messages = [{"role": "user", "content": "Question"}]
            system = "System prompt"

            result = generator._force_final_synthesis(messages, system)

            # Verify API was called
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args.kwargs
            # Tools should NOT be in the call
            assert 'tools' not in call_args
            assert 'tool_choice' not in call_args
            assert "answer" in result.lower()


@pytest.mark.unit
class TestSystemPrompt:
    """Test system prompt handling."""

    def test_system_prompt_contains_tool_instructions(self):
        """Test that the system prompt contains tool usage instructions."""
        assert "Search Tool Usage" in AIGenerator.SYSTEM_PROMPT
        assert "Outline Tool Usage" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_contains_sequential_tool_instructions(self):
        """Test that the system prompt mentions sequential tool calling."""
        assert "Sequential Tool Calling" in AIGenerator.SYSTEM_PROMPT
        assert "2 sequential tool calls" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_contains_response_protocol(self):
        """Test that the system prompt contains response protocol."""
        assert "Response Protocol" in AIGenerator.SYSTEM_PROMPT
        assert "General knowledge questions" in AIGenerator.SYSTEM_PROMPT
        assert "Course-specific questions" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_emphasizes_brevity(self):
        """Test that the system prompt emphasizes brevity."""
        assert "Brief" in AIGenerator.SYSTEM_PROMPT or "Concise" in AIGenerator.SYSTEM_PROMPT
        assert "No meta-commentary" in AIGenerator.SYSTEM_PROMPT


@pytest.mark.unit
class TestAIGeneratorErrors:
    """Test error handling in AIGenerator."""

    def test_api_key_error_propagates(self, mock_config):
        """Test that API key errors propagate up."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create = Mock(
                side_effect=Exception("401 Unauthorized: Invalid API key")
            )
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )
            generator.client = mock_client

            # Should raise exception
            with pytest.raises(Exception, match="401 Unauthorized"):
                generator.generate_response("Test query")

    def test_rate_limit_error_propagates(self, mock_config):
        """Test that rate limit errors propagate up."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create = Mock(
                side_effect=Exception("429 Too Many Requests")
            )
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )
            generator.client = mock_client

            with pytest.raises(Exception, match="Too Many Requests"):
                generator.generate_response("Test query")


@pytest.mark.unit
class TestExtractText:
    """Test the _extract_text helper method."""

    def test_extract_text_from_response(self, mock_config):
        """Test extracting text from a Claude response."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )

            response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Test response text"
            response.content = [mock_content]

            result = generator._extract_text(response)
            assert result == "Test response text"
