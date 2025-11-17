import streamlit as st
import openai
from datetime import datetime

# Dedicated RTSS agent helper
def generate_rtss_section(
    case_text: str,
    user_prompt: str,
    model: str,
    temperature: float,
    on_chunk=None,
) -> str:
    rtss_system_prompt = (
        "You are the RTSS Agent, specializing in Rehabilitation Treatment Specification System "
        "frameworks for speech-language pathology. Your sole task is to translate a given "
        "SLP case narrative into an RTSS-aligned plan that stays perfectly coherent with the "
        "provided case details."
    )
    rtss_user_prompt = f"""User request: {user_prompt}\n\nCase narrative: {case_text}\n\nCreate an RTSS section that includes: \n- Target (participation, impairment, or contextual focus) tied to the case goals.\n- Ingredients (specific clinician actions/techniques) with enough detail for another SLP to reproduce.\n- Mechanisms of action explaining why those ingredients are expected to work.\n- Dosage/parameters (session length, frequency, cues, materials).\n- Expected short-term markers of progress.\nFormat as concise markdown with clear subheadings. Do not contradict the narrative; infer only what is supported by it."""
    response_stream = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": rtss_system_prompt},
            {"role": "user", "content": rtss_user_prompt}
        ],
        temperature=min(max(temperature, 0.0), 1.0),
        max_tokens=800,
        stream=True
    )
    section_text = ""
    for chunk in response_stream:
        delta = chunk.choices[0].delta
        if not delta:
            continue
        delta_content = getattr(delta, "content", None)
        if delta_content is None and isinstance(delta, dict):
            delta_content = delta.get("content")
        if not delta_content:
            continue
        section_text += delta_content
        if on_chunk:
            on_chunk(section_text)
    return section_text.strip()

# Page configuration
st.set_page_config(
    page_title="Clinical Case Study Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4F46E5;
        margin-bottom: 0.5rem;
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
    }
    .user-message {
        background-color: #EEF2FF;
        border-left: 4px solid #4F46E5;
    }
    .assistant-message {
        background-color: #F9FAFB;
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)


def render_user_message(content: str) -> str:
    return f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{content}</div>'

def render_assistant_message(content: str) -> str:
    return f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong><br>{content}</div>'

# Get API key from secrets
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ OpenAI API key not found in secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rtss_enabled' not in st.session_state:
    st.session_state.rtss_enabled = False

# Sidebar
with st.sidebar:
    st.markdown("### 📖 Quick Prompts")
    
    quick_prompts = [
        "create a moderate Broca's aphasia case for learning initial assessment",
        "create a dementia case for treatment planning",
        "Demonstration of collaborative goal setting with conversation scripts",
        "Demonstration of motivational interviewing with conversation scripts",
        "Demonstration of a specific treatment technique with conversation scripts"
    ]
    dropdown_options = ["Select a quick prompt…"] + quick_prompts
    selected_prompt = st.selectbox("Choose a quick prompt", dropdown_options, index=0)
    if st.button("Send Quick Prompt", use_container_width=True, disabled=selected_prompt == dropdown_options[0]):
        st.session_state["pending_user_input"] = selected_prompt
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    difficulty = "Intermediate (Graduate)"
    include_assessment = True
    
    st.markdown("---")
    
    # Model selection
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=0,
        help="Select the OpenAI model to use"
    )
    
    # Temperature
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative, lower values more focused"
    )
    st.session_state.rtss_enabled = st.checkbox(
        "Generate RTSS section",
        value=st.session_state.get("rtss_enabled", False),
        help="Check to append a Rehabilitation Treatment Specification System plan after each case."
    )
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")

# Main content
st.markdown('<div class="main-header">📚 Clinical Case Study Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered case studies for Speech-Language Pathology education</div>', unsafe_allow_html=True)

live_user_placeholder = None
live_assistant_placeholder = None
conversation_container = st.container()
with conversation_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(render_user_message(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(render_assistant_message(message["content"]), unsafe_allow_html=True)
    live_user_placeholder = st.empty()
    live_assistant_placeholder = st.empty()

# Chat input (process queued quick prompts before showing input box)
queued_input = st.session_state.pop("pending_user_input", None)
user_input = st.chat_input("Describe the case study you'd like to create...")
pending_input = queued_input or user_input

if pending_input and live_user_placeholder and live_assistant_placeholder:
    st.session_state.messages.append({"role": "user", "content": pending_input})
    live_user_placeholder.markdown(render_user_message(pending_input), unsafe_allow_html=True)
    normalized_input = pending_input.lower()
    is_initial_assessment = "initial assessment" in normalized_input
    transfer_keywords = ["transfer", "new setting", "transition", "new facility", "new clinic","discharge to"]
    is_transfer_initial = is_initial_assessment and any(keyword in normalized_input for keyword in transfer_keywords)
    rtss_keywords = ["rtss", "rehabilitation treatment specification", "treatment specification"]
    rtss_requested = any(keyword in normalized_input for keyword in rtss_keywords)
    
    # Create system prompt based on settings
    difficulty_map = {
        "Beginner (Undergraduate)": "beginner",
        "Intermediate (Graduate)": "intermediate",
        "Advanced (Clinical Fellows)": "advanced"
    }
    guiding_focus = "Focus these questions on determining initial assessment priorities and selecting appropriate standardized and informal tools." if is_initial_assessment else "Use these questions to spark discussion about differential diagnosis, intervention planning, and clinical reasoning."
    if is_initial_assessment and not is_transfer_initial:
        language_sample_note = "Ensure the language sample reflects spontaneous speech observable prior to formal testing."
    else:
        language_sample_note = ""
    initial_assessment_guidance = ""
    if is_initial_assessment:
        if is_transfer_initial:
            initial_assessment_guidance = (
                "\nWhen the user signals an initial assessment during a transfer of care, summarize what prior SLPs accomplished, the patient's recovery trajectory, and documentation the new setting would inherit. Highlight unanswered assessment questions for the new site and clarify what still requires standardized and informal testing."
            )
        else:
            initial_assessment_guidance = (
                "\nWhen the user signals an initial assessment for a new onset with no prior SLP involvement, limit the case to referral information, medical history, observable behaviors, and collateral reports available before assessments are administered. Do not invent completed SLP assessment results yet; instead, describe what needs to be investigated and why."
            )
    
    system_prompt = f"""You are an expert clinical educator specializing in speech-language pathology. Your role is to help university instructors create high-quality, realistic case studies for their students.

Always craft a single cohesive narrative case study (paragraph style, not bullet points) that weaves together (and do not pre-answer or give instructions for the guiding questions here):
- Patient demographics and background
- Detailed medical history and etiology
- Presenting symptoms and characteristics
- Assessment results (formal and informal) with interpretation
- Clinical observations and differential diagnosis considerations
- Treatment recommendations, prognosis.

Keep the narrative observational and exploratory; reserve any prioritization, tool selection, or explicit answers for the Guiding Questions section only.

After the narrative, append two explicit sections:
1. Language Sample — provide a short quoted transcript (4–6 sentences) that captures the client's spontaneous speech. Match the cadence of this reference format: "Well the, uh, the little… the little cookers are spinning up there and she’s trying to wash the plates but the water’s all, all floofing out. And the boy, he’s, he’s grabbing the stool cause he wants the cookie. They’re having a good time, I think, and the mother doesn’t know the window is, is, uh, smiling. It’s pretty noisy in that room." Use it only as stylistic guidance; compose a fresh sample aligned with the case’s disorder-specific features every time so it reflects the symptoms described in the user input. {language_sample_note}
2. Guiding Questions — provide discussion questions instructors can use in class. {guiding_focus}

Keep the entire response focused on the case (no general tips). Adjust complexity to {difficulty_map[difficulty]} level, ensure clinical accuracy, and use professional terminology appropriate for graduate-level speech-language pathology education.{initial_assessment_guidance}"""
    
    # Prepare messages for API
    api_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} 
        for m in st.session_state.messages
    ]
    
    # Stream OpenAI API response so content appears as it arrives
    try:
        assistant_message = ""
        with st.spinner("Generating case study..."):
            response_stream = openai.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=2000,
                temperature=temperature,
                stream=True
            )
            for chunk in response_stream:
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                delta_content = getattr(delta, "content", None)
                if delta_content is None and isinstance(delta, dict):
                    delta_content = delta.get("content")
                if not delta_content:
                    continue
                assistant_message += delta_content
                live_assistant_placeholder.markdown(
                    render_assistant_message(assistant_message),
                    unsafe_allow_html=True
                )
        
        # Generate RTSS section with dedicated agent and append to response
        if st.session_state.get("rtss_enabled", False) or rtss_requested:
            try:
                with st.spinner("Synthesizing RTSS plan..."):
                    base_message = assistant_message
                    def update_rtss(section_partial: str) -> None:
                        live_assistant_placeholder.markdown(
                            render_assistant_message(
                                f"{base_message}\n\n### RTSS (Rehabilitation Treatment Specification System)\n{section_partial}"
                            ),
                            unsafe_allow_html=True
                        )
                    rtss_section = generate_rtss_section(
                        case_text=base_message,
                        user_prompt=pending_input,
                        model=model,
                        temperature=temperature,
                        on_chunk=update_rtss
                    )
                if rtss_section:
                    assistant_message = (
                        f"{base_message}\n\n### RTSS (Rehabilitation Treatment Specification System)\n{rtss_section}"
                    )
                    live_assistant_placeholder.markdown(
                        render_assistant_message(assistant_message),
                        unsafe_allow_html=True
                    )
            except Exception as rtss_error:
                st.warning(f"RTSS agent skipped due to error: {rtss_error}")
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please check your API configuration and try again.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>Powered by OpenAI • Built for SLP Educators</p>
    </div>
    """,
    unsafe_allow_html=True
)