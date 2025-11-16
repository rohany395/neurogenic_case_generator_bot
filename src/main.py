import streamlit as st
import openai
from datetime import datetime

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
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to the Clinical Case Study Generator! I specialize in creating detailed case studies for speech-language pathology conditions like aphasia, dysarthria, apraxia, and more. How can I help you create educational materials for your students today?"
        }
    ]

# Sidebar
with st.sidebar:
    st.markdown("### 📖 Quick Prompts")
    
    quick_prompts = [
        "Create a case study for Broca's aphasia",
    ]
    
    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True):
            # Queue prompt so main input handler treats it like regular user text
            st.session_state["pending_user_input"] = prompt
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Difficulty level
    difficulty = st.selectbox(
        "Difficulty Level",
        ["Beginner (Undergraduate)", "Intermediate (Graduate)", "Advanced (Clinical Fellows)"],
        index=1,
        help="Adjust the complexity of generated case studies"
    )
    
    # Include assessment questions
    include_assessment = st.checkbox(
        "Include Discussion Questions",
        value=True,
        help="Add learning objectives and discussion questions for students"
    )
    
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
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation cleared! How can I help you create a new case study?"
            }
        ]
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
    
    # Create system prompt based on settings
    difficulty_map = {
        "Beginner (Undergraduate)": "beginner",
        "Intermediate (Graduate)": "intermediate",
        "Advanced (Clinical Fellows)": "advanced"
    }
    
    system_prompt = f"""You are an expert clinical educator specializing in speech-language pathology and communication disorders. Your role is to help university instructors create high-quality, realistic case studies for their students.

Always craft a single cohesive narrative case study (paragraph style, not bullet points) that weaves together:
- Patient demographics and background
- Detailed medical history and etiology
- Presenting symptoms and characteristics
- Assessment results (formal and informal) with interpretation
- Clinical observations and differential diagnosis considerations
- Treatment recommendations, prognosis, and instructional goals

After the narrative, append two explicit sections:
1. Language Profile — summarize expressive/receptive abilities, pragmatics, cognition, literacy in a concise table or bullet list.
2. Guiding Questions — provide {"discussion" if include_assessment else "reflection"} questions instructors can use in class.

Keep the entire response focused on the case (no general tips). Adjust complexity to {difficulty_map[difficulty]} level, ensure clinical accuracy, and use professional terminology appropriate for graduate-level speech-language pathology education."""
    
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