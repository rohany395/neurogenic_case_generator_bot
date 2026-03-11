"""
Dedicated sub-agents that generate optional case-study sections:
  - Language Profile / Communication Observations
  - RTSS (Rehabilitation Treatment Specification System)

Each agent calls the shared stream_chat_chunks helper and accepts an
optional on_chunk callback for live streaming updates in the UI.
"""

from typing import Callable, Optional
from anthropic import Anthropic

from llm import stream_chat_chunks
from utils import strip_language_profile_heading


# --------------------------------------------------------------------------- #
# Language Profile Agent
# --------------------------------------------------------------------------- #

_LANGUAGE_PROFILE_SYSTEM = (
    "You are the Language Profile Agent for a speech-language pathology educator. "
    "Your only job is to translate a case narrative into a detailed language profile/"
    "communication observations summary."
)

_LANGUAGE_PROFILE_EXEMPLAR = (
    "Language Sample / Communication Observations\n"
    "Collected through conversational tasks, picture description, narrative generation, "
    "and functional exchanges.\n"
    "Verbal Expression\n"
    "Fluent, effortless speech with intact prosody.\n"
    "Frequent semantic paraphasias, circumlocutions, occasional phonemic paraphasias, "
    "and occasional neologisms.\n"
    "Sentences are grammatically intact but often empty in content or tangential.\n"
    "Awareness of errors is inconsistent; spontaneous repairs are limited.\n"
    "Auditory Comprehension\n"
    "Difficulty understanding multi-step directions.\n"
    "Reduced accuracy for moderately complex yes/no questions.\n"
    "Difficulty understanding conversational discourse without visual supports or "
    "contextual cues.\n"
    "Repetition\n"
    "Moderately–severely impaired for multisyllabic words and sentence-level stimuli.\n"
    "Naming\n"
    "Moderate impairment in confrontation naming with semantic substitutions and "
    "occasional perseveration."
)


def generate_language_profile_section(
    case_text: str,
    user_prompt: str,
    model_config: dict,
    temperature: float,
    anthropic_client: Optional[Anthropic],
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate a Language Profile section from the case narrative.

    Returns the finished section text (heading stripped).
    Calls on_chunk(partial_text) after each delta if provided.
    """
    user_message = (
        f"User request: {user_prompt}\n\n"
        f"Case narrative: {case_text}\n\n"
        "Produce a Language Profile / Communication Observations section modeled on "
        "the exemplar below. Match the structure (intro plus subsections such as "
        "Verbal Expression, Auditory Comprehension, Repetition, Naming, etc.), but "
        "tailor every content line to the provided case details. Keep it concise, "
        "clinically rich, and coherent with the narrative.\n\n"
        f"Exemplar:\n{_LANGUAGE_PROFILE_EXEMPLAR}"
    )
    messages = [
        {"role": "system", "content": _LANGUAGE_PROFILE_SYSTEM},
        {"role": "user",   "content": user_message},
    ]

    profile_text = ""
    for delta in stream_chat_chunks(
        model_config=model_config,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
        anthropic_client=anthropic_client,
    ):
        profile_text += delta
        if on_chunk:
            on_chunk(profile_text)

    return strip_language_profile_heading(profile_text.strip())


# --------------------------------------------------------------------------- #
# RTSS Agent
# --------------------------------------------------------------------------- #

_RTSS_SYSTEM = (
    "You are the RTSS Agent, specialising in Rehabilitation Treatment Specification "
    "System frameworks for speech-language pathology. Your sole task is to translate "
    "a given SLP case narrative into an RTSS-aligned plan that stays perfectly "
    "coherent with the provided case details."
)


def generate_rtss_section(
    case_text: str,
    user_prompt: str,
    model_config: dict,
    temperature: float,
    anthropic_client: Optional[Anthropic],
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate an RTSS section from the case narrative.

    Returns the finished section text.
    Calls on_chunk(partial_text) after each delta if provided.
    """
    user_message = (
        f"User request: {user_prompt}\n\n"
        f"Case narrative: {case_text}\n\n"
        "Create an RTSS section that includes:\n"
        "- Target (participation, impairment, or contextual focus) tied to the case goals.\n"
        "- Ingredients (specific clinician actions/techniques) with enough detail for "
        "another SLP to reproduce.\n"
        "- Mechanisms of action explaining why those ingredients are expected to work.\n"
        "- Dosage/parameters (session length, frequency, cues, materials).\n"
        "- Expected short-term markers of progress.\n"
        "Format as concise markdown with clear subheadings. Do not contradict the "
        "narrative; infer only what is supported by it."
    )
    messages = [
        {"role": "system", "content": _RTSS_SYSTEM},
        {"role": "user",   "content": user_message},
    ]

    section_text = ""
    for delta in stream_chat_chunks(
        model_config=model_config,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
        anthropic_client=anthropic_client,
    ):
        section_text += delta
        if on_chunk:
            on_chunk(section_text)

    return section_text.strip()