# 📚 Clinical Case Study Generator

An AI-powered educational tool for generating realistic clinical case studies in speech-language pathology (SLP), specifically focused on neurogenic communication disorders.

## Overview

This Streamlit application helps university instructors create high-quality, clinically accurate case studies for teaching graduate-level SLP students. The system uses advanced LLMs (GPT-4o and Claude Sonnet 4) combined with RAG (Retrieval-Augmented Generation) to produce consistent, educationally valuable content.

## Features

- **Dual LLM Support**: Choose between OpenAI GPT-4o and Claude Sonnet 4
- **RAG-Enhanced Generation**: Uses curated reference cases for consistent style and quality
- **Context-Aware Output**: Automatically adapts for initial assessment vs. treatment planning scenarios
- **Modular Agents**: Optional Language Profile and RTSS (Rehabilitation Treatment Specification System) sections
- **Real-Time Streaming**: Responses appear token-by-token for immediate feedback
- **Copy to Clipboard**: Easy export of generated content

## System Architecture

```
User Input → Context Detection → RAG Retrieval → LLM Generation → Output
```

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohany395/neurogenic_case_generator_bot.git
   cd neurogenic_case_generator_bot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "sk-your-openai-key-here"
   ANTHROPIC_API_KEY = "sk-ant-your-anthropic-key-here"  # Optional
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

```
neurogenic_case_generator_bot/
├── .streamlit/
│   └── secrets.toml          # API keys (not in repo)
├── data/
│   └── chroma_db/            # Vector database (auto-generated)
├── ref_documents/            # Reference case studies (.docx)
├── src/
│   └── main.py               # Main application code
├── streamlit_app.py          # Entry point
├── requirements.txt          # Python dependencies
└── README.md
```

## Usage

### Basic Usage

1. Type a case description in the chat input:
   ```
   Create a moderate Broca's aphasia case for initial assessment
   ```

2. The system will generate:
   - A cohesive case narrative
   - A disorder-specific language sample
   - Guiding discussion questions

### Quick Prompts

Use the sidebar dropdown for pre-configured prompts:
- Broca's aphasia case for initial assessment
- Dementia case for treatment planning
- Collaborative goal setting demonstration
- Motivational interviewing demonstration

### Optional Sections

Enable via sidebar checkboxes:
- **Language Profile**: Detailed communication observations
- **RTSS Section**: Rehabilitation Treatment Specification System plan

### Context Detection

The system automatically adjusts output based on keywords:

| Keywords | Effect |
|----------|--------|
| "initial assessment" | Limits case to pre-evaluation information |
| "transfer", "new setting" | Includes prior SLP work and recovery trajectory |
| "rtss" | Triggers RTSS section generation |
| "language profile" | Triggers Language Profile generation |

## Technical Details

### Technologies Used

| Component | Technology |
|-----------|------------|
| Web Framework | Streamlit |
| LLMs | OpenAI GPT-4o, Claude Sonnet 4 |
| Vector Database | ChromaDB |
| Embeddings | OpenAI text-embedding |
| RAG Framework | LangChain |
| Document Parsing | python-docx |

### Key Functions

- `stream_chat_chunks()`: Unified streaming interface for OpenAI/Anthropic
- `get_best_matching_exemplar()`: Semantic search with keyword fallback
- `ingest_exemplars_to_vector_db()`: Index reference documents
- `generate_language_profile_section()`: Dedicated Language Profile agent
- `generate_rtss_section()`: Dedicated RTSS agent

## Configuration

### Settings (Sidebar)

- **Model**: Select between GPT-4o and Claude Sonnet 4
- **Creativity**: Temperature slider (0.0 - 1.0)
- **Clear Conversation**: Reset chat history

### Reference Documents

Place `.docx` case studies in `ref_documents/` folder. The system will automatically index them on first run for semantic search.

## Troubleshooting

### SQLite Version Error

If you encounter ChromaDB SQLite errors, the app includes an automatic fix using `pysqlite3-binary`.

### API Key Issues

Ensure your API keys are correctly configured in `.streamlit/secrets.toml`. The app will display an error message if keys are missing for the selected model.

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed as an educational tool for speech-language pathology programs to enhance clinical case-based learning.
