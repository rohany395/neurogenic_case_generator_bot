# Headings
LANGUAGE_PROFILE_HEADING_TEXT = "Language Profile / Communication Observations"
LANGUAGE_PROFILE_HEADING = f"### {LANGUAGE_PROFILE_HEADING_TEXT}"


# RAG / storage paths
REF_DOCUMENTS_DIR = "ref_documents"
CHROMA_DB_DIR = "data/chroma_db"


# LLM / model options
MODEL_OPTIONS: dict[str, dict] = {
    "OpenAI GPT-4o":    {"provider": "openai",     "id": "gpt-4o"},
    "Claude Sonnet 4":  {"provider": "anthropic",   "id": "claude-sonnet-4-20250514"},
}


# Case generation
ETHNICITIES: list[str] = [
    "African American",
    "Caucasian/White",
    "Hispanic/Latino",
    "Asian American",
    "Native American",
    "Pacific Islander",
    "Middle Eastern",
    "South Asian",
    "East Asian",
    "Southeast Asian",
    "Caribbean",
    "Mixed race/Multiracial",
]

NEW_CASE_KEYWORDS: list[str] = [
    "create", "generate", "make", "new case", "another case", "different case",
    "case study", "case for", "demonstration of", "show me", "give me",
    "broca", "wernicke", "aphasia", "dementia", "dysarthria", "apraxia",
    "tbi", "traumatic brain", "stroke", "parkinsons", "parkinson's", "als",
    "primary progressive", "right hemisphere", "cognitive-communication",
]

TRANSFER_KEYWORDS: list[str] = [
    "transfer", "new setting", "transition", "new facility", "new clinic", "discharge to",
]

RTSS_KEYWORDS: list[str] = [
    "rtss", "rehabilitation treatment specification", "treatment specification",
]

LANGUAGE_PROFILE_KEYWORDS: list[str] = [
    "language profile", "communication observations", "verbal expression profile",
]

# Quick prompts
QUICK_PROMPTS: list[dict] = [
    {"text": "Create a moderate Broca's aphasia case for learning initial assessment"},
    {"text": "Create a dementia case for treatment planning"},
    {"text": "Demonstration of collaborative goal setting with conversation scripts"},
    {"text": "Demonstration of motivational interviewing with conversation scripts"},
    {"text": "Demonstration of a specific treatment technique with conversation scripts"},
    {"text": "Create a case of flaccid dysarthria for treatment planning"},
]

# Difficulty levels
DIFFICULTY_MAP: dict[str, str] = {
    "Beginner (Undergraduate)": "beginner",
    "Intermediate (Graduate)":  "intermediate",
    "Advanced (Clinical Fellows)": "advanced",
}