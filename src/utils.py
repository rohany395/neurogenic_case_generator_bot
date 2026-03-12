"""
Stateless helper functions used across the application.
"""

from config import (
    NEW_CASE_KEYWORDS,
    LANGUAGE_PROFILE_HEADING_TEXT,
)


def is_new_case_request(user_input: str, messages: list) -> bool:
    """Return True if the user message looks like a brand-new case request."""
    normalized = user_input.lower()

    # First message is always a new case
    if len(messages) <= 1:
        return True

    for keyword in NEW_CASE_KEYWORDS:
        if keyword in normalized:
            return True

    return False


def strip_language_profile_heading(text: str) -> str:
    """Remove the Language Profile heading line (and blank lines after it) from text."""
    if not text:
        return ""
    lines = text.lstrip().splitlines()
    while lines:
        first_line = lines[0].strip()
        if LANGUAGE_PROFILE_HEADING_TEXT.lower() in first_line.lower():
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        else:
            break
    return "\n".join(lines).strip()