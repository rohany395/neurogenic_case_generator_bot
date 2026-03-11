"""
Streamlit application entry-point.
Handles only UI layout, session state, and the request/response loop.
All heavy logic is delegated to the sibling modules.
"""

# ── SQLite shim (must run before any ChromaDB import) ────────────────────────
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import random
from typing import Optional

import openai
import streamlit as st
from anthropic import Anthropic

# Local modules
from config import (
    MODEL_OPTIONS,
    QUICK_PROMPTS,
    ETHNICITIES,
    TRANSFER_KEYWORDS,
    RTSS_KEYWORDS,
    LANGUAGE_PROFILE_KEYWORDS,
    LANGUAGE_PROFILE_HEADING,
)
from agents import generate_language_profile_section, generate_rtss_section
from llm import stream_chat_chunks
from prompts import build_system_prompt
from rag import get_best_matching_exemplar
from ui import (
    inject_css,
    render_user_message,
    render_assistant_message,
    render_copy_button,
)
from utils import is_new_case_request, strip_language_profile_heading

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Case Study Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── API clients ──────────────────────────────────────────────────────────────

openai_api_key: Optional[str] = st.secrets.get("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

anthropic_api_key: Optional[str] = st.secrets.get("ANTHROPIC_API_KEY")
anthropic_client: Optional[Anthropic] = (
    Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
)

# ── Session state ─────────────────────────────────────────────────────────────

st.session_state.setdefault("messages", [])
st.session_state.setdefault("rtss_enabled", False)
st.session_state.setdefault("language_profile_enabled", False)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ➕ Additional Content")
    st.session_state.language_profile_enabled = st.checkbox(
        "Generate Language Profile section",
        value=st.session_state.language_profile_enabled,
        help=(
            "Append a detailed Language Profile / Communication Observations summary "
            "modelled on clinical documentation."
        ),
    )
    st.session_state.rtss_enabled = st.checkbox(
        "Generate RTSS section",
        value=st.session_state.rtss_enabled,
        help="Append a Rehabilitation Treatment Specification System plan after each case.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    selected_model_label = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Choose between the default OpenAI model and the integrated Claude model",
    )
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative, lower values more focused",
    )
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

model_config = MODEL_OPTIONS[selected_model_label]

# Guard: validate API keys for the chosen provider
if model_config["provider"] == "openai" and not openai_api_key:
    st.error("⚠️ OpenAI model selected but OPENAI_API_KEY is not configured.")
    st.stop()
if model_config["provider"] == "anthropic" and not anthropic_client:
    st.error("⚠️ Claude model selected but ANTHROPIC_API_KEY is not configured.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">📚 Clinical Case Study Generator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center;color:#6B7280;padding:0;'>"
    "<p>An AI-powered educational case study generator for neurogenic communication disorders</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Quick-prompt tiles (shown only on empty conversation) ─────────────────────

if not st.session_state.messages:
    st.markdown(
        "<div class='welcome-section'>"
        "<h3>👋 Welcome! Get started with a quick prompt:</h3>"
        "<p>Select one of the options below or type your own request</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    for idx, prompt_data in enumerate(QUICK_PROMPTS):
        col = col1 if idx % 2 == 0 else col2
        with col:
            if st.button(
                prompt_data["text"],
                key=f"quick_prompt_{idx}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state["pending_user_input"] = prompt_data["text"]
                st.rerun()

# ── Conversation history ──────────────────────────────────────────────────────

conversation_container = st.container()
with conversation_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(render_user_message(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                render_assistant_message(message["content"]), unsafe_allow_html=True
            )
            render_copy_button(message["content"])

    live_user_placeholder = st.empty()
    live_assistant_placeholder = st.empty()

# ── Input handling ────────────────────────────────────────────────────────────

queued_input = st.session_state.pop("pending_user_input", None)
user_input = st.chat_input("Describe the case study you'd like to create...")
pending_input = queued_input or user_input

if not pending_input:
    st.stop()

# ── Process user request ──────────────────────────────────────────────────────

st.session_state.messages.append({"role": "user", "content": pending_input})
live_user_placeholder.markdown(
    render_user_message(pending_input), unsafe_allow_html=True
)

normalized_input = pending_input.lower()

is_initial_assessment = "initial assessment" in normalized_input
is_transfer_initial = is_initial_assessment and any(
    kw in normalized_input for kw in TRANSFER_KEYWORDS
)
rtss_requested = any(kw in normalized_input for kw in RTSS_KEYWORDS)
language_profile_requested = any(kw in normalized_input for kw in LANGUAGE_PROFILE_KEYWORDS)

# Ethnicity randomisation for new cases only
if is_new_case_request(pending_input, st.session_state.messages):
    selected_ethnicity = random.choice(ETHNICITIES)
else:
    selected_ethnicity = ""

reference_content, _ = get_best_matching_exemplar(pending_input, openai_api_key)

system_prompt = build_system_prompt(
    difficulty="Intermediate (Graduate)",
    is_initial_assessment=is_initial_assessment,
    is_transfer_initial=is_transfer_initial,
    reference_content=reference_content,
    ethnicity=selected_ethnicity,
)

api_messages = [{"role": "system", "content": system_prompt}] + [
    {"role": m["role"], "content": m["content"]}
    for m in st.session_state.messages
]

# ── Stream main response ──────────────────────────────────────────────────────

try:
    assistant_message = ""
    with st.spinner("Generating case study..."):
        for delta in stream_chat_chunks(
            model_config=model_config,
            messages=api_messages,
            temperature=temperature,
            max_tokens=2000,
            anthropic_client=anthropic_client,
        ):
            assistant_message += delta
            live_assistant_placeholder.markdown(
                render_assistant_message(assistant_message), unsafe_allow_html=True
            )

    base_message = assistant_message

    # ── Language Profile agent ────────────────────────────────────────────────

    if st.session_state.language_profile_enabled or language_profile_requested:
        try:
            with st.spinner("Drafting language profile..."):
                def _on_lp_chunk(partial: str) -> None:
                    sanitized = strip_language_profile_heading(partial)
                    live_assistant_placeholder.markdown(
                        render_assistant_message(
                            f"{base_message}\n\n{LANGUAGE_PROFILE_HEADING}\n{sanitized}"
                        ),
                        unsafe_allow_html=True,
                    )

                lp_section = generate_language_profile_section(
                    case_text=base_message,
                    user_prompt=pending_input,
                    model_config=model_config,
                    temperature=temperature,
                    anthropic_client=anthropic_client,
                    on_chunk=_on_lp_chunk,
                )

            if lp_section:
                assistant_message = (
                    f"{base_message}\n\n{LANGUAGE_PROFILE_HEADING}\n{lp_section}"
                )
                live_assistant_placeholder.markdown(
                    render_assistant_message(assistant_message), unsafe_allow_html=True
                )
                base_message = assistant_message
        except Exception as lp_err:
            st.warning(f"Language profile agent skipped: {lp_err}")

    # ── RTSS agent ────────────────────────────────────────────────────────────

    if st.session_state.rtss_enabled or rtss_requested:
        try:
            with st.spinner("Synthesising RTSS plan..."):
                def _on_rtss_chunk(partial: str) -> None:
                    live_assistant_placeholder.markdown(
                        render_assistant_message(f"{base_message}\n\n{partial}"),
                        unsafe_allow_html=True,
                    )

                rtss_section = generate_rtss_section(
                    case_text=base_message,
                    user_prompt=pending_input,
                    model_config=model_config,
                    temperature=temperature,
                    anthropic_client=anthropic_client,
                    on_chunk=_on_rtss_chunk,
                )

            if rtss_section:
                assistant_message = f"{base_message}\n\n{rtss_section}"
                live_assistant_placeholder.markdown(
                    render_assistant_message(assistant_message), unsafe_allow_html=True
                )
        except Exception as rtss_err:
            st.warning(f"RTSS agent skipped: {rtss_err}")

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    st.rerun()

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Please check your API configuration and try again.")