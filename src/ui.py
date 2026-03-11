"""
ui.py
~~~~~
Streamlit UI helpers: CSS injection, message renderers, and the
copy-to-clipboard button component.
"""

import html
import json

import streamlit as st
import streamlit.components.v1 as components


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #

APP_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4F46E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #1F2937;
        position: relative;
    }
    .user-message {
        background-color: #EEF2FF;
        border-left: 4px solid #4F46E5;
    }
    .assistant-message {
        background-color: #F9FAFB;
        border-left: 4px solid #10B981;
    }
    /* Force dark text in chat messages regardless of theme */
    .chat-message * { color: #1F2937 !important; }
    .chat-message strong { color: #111827 !important; }

    /* Quick prompt tiles */
    .quick-prompts-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        padding: 2rem 1rem;
        max-width: 900px;
        margin: 0 auto;
    }
    .quick-prompt-tile {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border: 1px solid #C7D2FE;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        flex: 1 1 calc(50% - 1rem);
        min-width: 250px;
        max-width: 400px;
        text-align: left;
    }
    .quick-prompt-tile:hover {
        background: linear-gradient(135deg, #E0E7FF 0%, #C7D2FE 100%);
        border-color: #A5B4FC;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
    }
    .quick-prompt-tile p {
        margin: 0;
        color: #3730A3 !important;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .quick-prompt-tile .tile-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .welcome-section {
        text-align: center;
        padding: 2rem 1rem;
        color: #6B7280;
    }
    .welcome-section h3 {
        color: #374151;
        margin-bottom: 0.5rem;
    }
</style>
"""


def inject_css() -> None:
    """Inject application-wide CSS into the Streamlit page."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Message renderers
# --------------------------------------------------------------------------- #

def render_user_message(content: str) -> str:
    return (
        f'<div class="chat-message user-message">'
        f"<strong>👤 You:</strong><br>{content}"
        f"</div>"
    )


def render_assistant_message(content: str) -> str:
    return (
        f'<div class="chat-message assistant-message">'
        f"<strong>🤖 Assistant:</strong><br>{content}"
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# Copy-to-clipboard button
# --------------------------------------------------------------------------- #

def render_copy_button(content: str) -> None:
    """Render a functional clipboard copy button via an inline HTML component."""
    js_string = json.dumps(content)
    escaped_js_string = html.escape(js_string)
    html_code = f"""
    <button id="copyBtn" style="
        background-color: #E5E7EB;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
        font-size: 0.8rem;
        color: #374151;
        transition: background-color 0.2s;
    "
    onmouseover="this.style.backgroundColor='#D1D5DB'"
    onmouseout="this.style.backgroundColor='#E5E7EB'"
    onclick="
        var text = {escaped_js_string};
        var btn = document.getElementById('copyBtn');
        function showCopied() {{
            btn.textContent = '✓ Copied!';
            setTimeout(function() {{ btn.textContent = '📋 Copy'; }}, 2000);
        }}
        navigator.clipboard.writeText(text).then(showCopied).catch(function(err) {{
            var textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            try {{
                document.execCommand('copy');
                showCopied();
            }} catch(e) {{
                console.error('Copy failed:', e);
            }}
            document.body.removeChild(textarea);
        }});
    ">📋 Copy</button>
    """
    components.html(html_code, height=35)