import anthropic
from typing import List, Optional, Dict, Tuple


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You can make up to 2 sequential tool calls to gather information before answering
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Outline Tool Usage:
- Use the outline tool for questions about course structure, curriculum, or "what does X cover"
- Examples: "What does the MCP course cover?", "What lessons are in the Introduction course?", "Show me the outline for X"
- Returns course title, link, and complete lesson list

Sequential Tool Calling:
- For complex questions, you may call multiple tools in sequence
- Example: First get an outline to find a lesson title, then search for content about that topic
- Each tool call gives you new information to use in subsequent calls or your final answer
- Maximum 2 tool rounds per query - use them wisely

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search or use outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports sequential tool calling - Claude can make up to max_tool_rounds
        tool calls in separate API rounds before providing the final answer.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Build initial messages list with user query
        messages = [{"role": "user", "content": query}]

        # If no tools available, make single API call and return
        if not tools or not tool_manager:
            response = self.client.messages.create(
                **self.base_params, messages=messages, system=system_content
            )
            return response.content[0].text

        # === MAIN LOOP: Sequential tool calling ===
        # Each iteration: API call -> check for tool use -> execute tools -> repeat
        for round_num in range(self.max_tool_rounds + 1):

            # Prepare API parameters for this round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Append assistant's response to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # TERMINATION 1: No tool use requested - Claude answered directly
            if response.stop_reason != "tool_use":
                return self._extract_text(response)

            # TERMINATION 2: Max rounds reached - force final synthesis
            if round_num >= self.max_tool_rounds:
                return self._force_final_synthesis(messages, system_content)

            # Execute tools and build result blocks
            tool_results, had_error = self._execute_tools_from_response(response, tool_manager)

            # Append tool results to conversation for next round
            messages.append({"role": "user", "content": tool_results})

            # TERMINATION 3: Tool execution failed - force final synthesis with error info
            if had_error:
                return self._force_final_synthesis(messages, system_content)

            # Loop continues - tools remain available for next round

        # Should not reach here, but return safe default
        return "I apologize, but I was unable to complete your request."

    def _execute_tools_from_response(self, response, tool_manager) -> tuple:
        """
        Execute all tools in a Claude response.

        Args:
            response: Claude API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results list, had_error boolean)
        """
        tool_results = []
        had_error = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Tool execution failed - include error in results
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error executing {content_block.name}: {str(e)}",
                            "is_error": True,
                        }
                    )
                    had_error = True

        return tool_results, had_error

    def _force_final_synthesis(self, messages: List[Dict], system_content: str) -> str:
        """
        Make a final API call without tools to force Claude to synthesize an answer.

        Args:
            messages: Current conversation history
            system_content: System prompt

        Returns:
            Final text response from Claude
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # Note: tools and tool_choice intentionally omitted
        }

        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response)

    def _extract_text(self, response) -> str:
        """
        Extract text content from a Claude API response.

        Args:
            response: Claude API response

        Returns:
            Text content of the response
        """
        return response.content[0].text
