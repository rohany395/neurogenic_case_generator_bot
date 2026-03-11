"""
Low-level LLM helpers: message format conversion and the unified
streaming generator that works for both OpenAI and Anthropic.
"""

from typing import Optional, Generator
import openai
from anthropic import Anthropic



# Message format helpers
def convert_messages_for_anthropic(
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """
    Split a mixed message list into:
      - system_prompt  (str)  – concatenated content of all system messages
      - convo_messages (list) – user/assistant turns only
    """
    system_parts: list[str] = []
    convo_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        elif role in ("user", "assistant"):
            convo_messages.append({"role": role, "content": content})

    return "\n\n".join(system_parts), convo_messages


# Unified streaming generator
def stream_chat_chunks(
    model_config: dict,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    anthropic_client: Optional[Anthropic],
) -> Generator[str, None, None]:
    """
    Yield text delta chunks from whichever provider is specified in model_config.

    model_config must contain:
      - "provider": "openai" | "anthropic"
      - "id":       model identifier string
    """
    provider = model_config["provider"]
    model_id = model_config["id"]
    clamped_temp = min(max(temperature, 0.0), 1.0)

    if provider == "openai":
        response_stream = openai.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=clamped_temp,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if not delta:
                continue
            delta_content = getattr(delta, "content", None)
            if delta_content is None and isinstance(delta, dict):
                delta_content = delta.get("content")
            if delta_content:
                yield delta_content
        return

    # --- Anthropic ---
    if not anthropic_client:
        raise RuntimeError(
            "Claude model selected but Anthropic API key is missing."
        )
    system_prompt, anthropic_messages = convert_messages_for_anthropic(messages)
    with anthropic_client.messages.stream(
        model=model_id,
        max_tokens=max_tokens,
        temperature=clamped_temp,
        system=system_prompt or "",
        messages=anthropic_messages,
    ) as stream:
        for event in stream:
            if (
                event.type == "content_block_delta"
                and getattr(event.delta, "type", "") == "text_delta"
            ):
                yield event.delta.text