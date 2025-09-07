"""LiteLLM model provider.

- Docs: https://docs.litellm.ai/
"""

import json
import logging
from typing import Any, AsyncGenerator, Optional, Type, TypedDict, TypeVar, Union, cast

from pydantic import BaseModel
from typing_extensions import Unpack, override

from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec
from strands.models.openai import OpenAIModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIModelNonStreaming(OpenAIModel):
    """OpenAIModel Non-Streaming model provider implementation."""
    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the OpenAI model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("request=<%s>", request)

        # Check if streaming is disabled in the params
        config = self.get_config()
        params = config.get("params") or {}
        #is_streaming = params.get("stream", True)
        is_streaming = False

        #litellm_request = {**request}

        #litellm_request["stream"] = is_streaming

        logger.debug("invoking model with stream=%s", is_streaming)

        if not is_streaming:
            response = await self.client.chat.completions.create(**request)

            logger.debug("got non-streaming response from model")
            yield self.format_chunk({"chunk_type": "message_start"})
            yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

            tool_calls: dict[int, list[Any]] = {}
            finish_reason = None

            if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
                choice = response.choices[0]

                if hasattr(choice, "message") and choice.message:
                    if hasattr(choice.message, "content") and choice.message.content:
                        yield self.format_chunk(
                            {"chunk_type": "content_delta", "data_type": "text", "data": choice.message.content}
                        )

                    if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "reasoning_content",
                                "data": choice.message.reasoning_content,
                            }
                        )

                    if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                        for i, tool_call in enumerate(choice.message.tool_calls):
                            tool_calls.setdefault(i, []).append(tool_call)

                if hasattr(choice, "finish_reason"):
                    finish_reason = choice.finish_reason

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

            for tool_deltas in tool_calls.values():
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

                for tool_delta in tool_deltas:
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

            # Add usage information if available
            if hasattr(response, "usage"):
                yield self.format_chunk({"chunk_type": "metadata", "data": response.usage})
        else:
            # For streaming, use the streaming API
            response = await self.client.chat.completions.create(**request)

            logger.debug("got streaming response from model")
            yield self.format_chunk({"chunk_type": "message_start"})
            yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

            streaming_tool_calls: dict[int, list[Any]] = {}
            finish_reason = None

            try:
                async for event in response:
                    # Defensive: skip events with empty or missing choices
                    if not getattr(event, "choices", None):
                        continue
                    choice = event.choices[0]

                    if choice.delta.content:
                        yield self.format_chunk(
                            {"chunk_type": "content_delta", "data_type": "text", "data": choice.delta.content}
                        )

                    if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "reasoning_content",
                                "data": choice.delta.reasoning_content,
                            }
                        )

                    for tool_call in choice.delta.tool_calls or []:
                        streaming_tool_calls.setdefault(tool_call.index, []).append(tool_call)

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        break
            except Exception as e:
                logger.warning("Error processing streaming response: %s", e)

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

            # Process tool calls
            for tool_deltas in streaming_tool_calls.values():
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

                for tool_delta in tool_deltas:
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

            try:
                last_event = None
                async for event in response:
                    last_event = event

                # Use the last event for usage information
                if last_event and hasattr(last_event, "usage"):
                    yield self.format_chunk({"chunk_type": "metadata", "data": last_event.usage})
            except Exception:
                # If there's an error collecting remaining events, just continue
                pass

        logger.debug("finished processing response from model")

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format an OpenAI compatible chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An OpenAI compatible chat streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to an OpenAI-compatible
                format.
        """
        return {
            "messages": self.format_request_messages(messages, system_prompt),
            "model": self.config["model_id"],
            "stream": False,
            #"stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **cast(dict[str, Any], self.config.get("params", {})),
        }