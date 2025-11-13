import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, TypedDict
from pathlib import Path
from io import BytesIO

import streamlit as st
from openai import OpenAI
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import docx
except ImportError:
    docx = None

# =========================
# UI + API initialization
# =========================
st.set_page_config(
    page_title="Neurogenic Communication Case Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🧠 Neurogenic Communication Case Generator")

openai_key = st.secrets.get("API_KEY")
if not openai_key:
    st.error("❌ Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=openai_key)
MODEL_ID = "gpt-4.1"

# =========================
# Reference data loading
# =========================
REF_ROOT = Path("reference_data")
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def load_reference_data() -> Dict[str, Any]:
    lesion_symptom = _read_json(REF_ROOT / "clinical" / "lesion_symptom_map.json", {
        "Broca's aphasia": {"lesion": ["left IFG", "MCA superior"], "key_signs": ["nonfluent", "agrammatism", "impaired repetition"]},
        "AOS": {"lesion": ["left premotor/insula"], "key_signs": ["groping", "inconsistent errors", "prosody disrupted"]},
        "PD hypokinetic dysarthria": {"lesion": ["basal ganglia circuit"], "key_signs": ["reduced loudness", "monopitch", "short rushes"]}
    })
    onset_course = _read_json(REF_ROOT / "clinical" / "onset_course_table.json", {
        "stroke": {"onset": ["acute", "subacute"], "course": "stepwise → gradual improvement"},
        "PD": {"onset": ["chronic"], "course": "progressive"}
    })
    test_norms = _read_json(REF_ROOT / "clinical" / "test_norms.json", {
        "WAB_AQ": {"mild": "76–93", "moderate": "51–75", "severe": "<=50"},
        "DDK": {"AMR_ba": "4–7 syl/s typical adult", "SMR_pataka": "3–6 syl/s typical adult"}
    })

    learning_goal_map = _read_json(REF_ROOT / "pedagogy" / "learning_goal_map.json", {
        "novice": {"case_complexity": "low", "ambiguity": "low", "focus": ["impairment-level"]},
        "intermediate": {"case_complexity": "moderate", "ambiguity": "moderate", "focus": ["activity", "impairment"]},
        "advanced": {"case_complexity": "high", "ambiguity": "moderate-high", "focus": ["participation", "activity", "impairment"]}
    })
    reasoning_rubrics = _read_json(REF_ROOT / "pedagogy" / "reasoning_rubrics.json", {
        "novice": ["identifies signs", "names likely disorder"],
        "advanced": ["integrates PCC", "justifies targets with mechanisms"]
    })
    icf_examples = _read_json(REF_ROOT / "pedagogy" / "icf_examples.json", {
        "goals": ["order coffee in a busy café", "video call with family", "participate in team meeting"],
        "supports": ["spouse coaching", "visual keyword boards"],
        "barriers": ["noise", "fatigue", "time pressure"]
    })
    mi_cheatsheet = _read_json(REF_ROOT / "pedagogy" / "mi_cheatsheet.json", {
        "oars": ["Open questions", "Affirmations", "Reflections", "Summaries"],
        "change_talk_cues": ["desire", "ability", "reasons", "need", "commitment", "activation", "taking steps"],
        "examples": {
            "open_question": "What matters most for your conversations at home?",
            "complex_reflection": "It sounds like the phone calls are tiring, yet staying connected is really important to you."
        }
    })
    gas_templates = _read_json(REF_ROOT / "pedagogy" / "gas_templates.json", {
        "scale_labels": ["-2","-1","0","+1","+2"],
        "anchor_rules": "Anchor 0 = expected level after planned intervention; baseline mapped to -1 or -2; anchors observable and context-bound.",
        "example": {
            "goal": "Participate in a 5-minute phone call with daughter, initiating at least 3 turns with <10% word-finding breakdowns using strategies.",
            "anchors": {
                "-2": "Needs caregiver to lead; <1 initiated turn; frequent breakdowns; no strategy use.",
                "-1": "Initiates 1–2 turns with prompts; inconsistent strategy use.",
                "0": "Initiates 3 turns with minimal prompts; consistent cueing; ≤10% breakdowns.",
                "+1": "Initiates 4–5 turns independently; rare breakdowns.",
                "+2": "Sustains 6+ turns independently with strategy carryover."
            },
            "measurement": {"tool":"CIU/min + partner rating","cadence":["baseline","week4","discharge"]}
        }
    })

    treatment_protocols = _read_json(REF_ROOT / "rtss" / "treatment_protocols.json", {
        "SFA": {"mechanism":"feature cueing → semantic network activation","typical_dose":{"session_min":45,"sessions_per_week":3,"total_weeks":4}},
        "VNeST": {"mechanism":"verb-centered schema retrieval","typical_dose":{"session_min":60,"sessions_per_week":2,"total_weeks":6}},
        "MIT": {"mechanism":"right-hemisphere melodic-prosodic recruitment","typical_dose":{"session_min":60,"sessions_per_week":3,"total_weeks":6}}
    })
    measure_registry = _read_json(REF_ROOT / "rtss" / "measure_registry.json", {
        "CIU/min": {"construct":"discourse informativeness","direction":"higher=better","cadence":["baseline","week4"]},
        "CPIB": {"construct":"participation impact","direction":"higher=better","cadence":["baseline","week4","discharge"]},
        "SIT": {"construct":"intelligibility","direction":"higher=better","cadence":["baseline","week4"]},
        "GAS": {"construct":"goal attainment","direction":"higher=better","cadence":["baseline","mid","discharge"]}
    })
    extracted_dir = REF_ROOT / "rtss" / "extracted_evidence"
    extracted = []
    if extracted_dir.exists():
        for p in extracted_dir.glob("*.json"):
            try:
                extracted.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue

    return {
        "clinical": {"lesion_symptom": lesion_symptom, "onset_course": onset_course, "test_norms": test_norms},
        "pedagogy": {
            "learning_goal_map": learning_goal_map,
            "reasoning_rubrics": reasoning_rubrics,
            "icf_examples": icf_examples,
            "mi_cheatsheet": mi_cheatsheet,
            "gas_templates": gas_templates
        },
        "rtss": {"protocols": treatment_protocols, "measure_registry": measure_registry, "evidence": extracted}
    }

REF = load_reference_data()

# =========================
# Upload helpers 
# =========================
MAX_RESOURCE_CHARS = 8000

def _preview_text(s: str, n: int = 300) -> str:
    s = s.strip()
    return (s[:n] + "…") if len(s) > n else s

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    if PyPDF2 is None:
        return "Error: PyPDF2 not installed. Install with: pip install PyPDF2"
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text_parts = []
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())
        return "\n".join(text_parts)
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file) -> str:
    """Extract text from Word document"""
    if docx is None:
        return "Error: python-docx not installed. Install with: pip install python-docx"
    
    try:
        doc = docx.Document(file)
        text_parts = []
        for paragraph in doc.paragraphs:
            text_parts.append(paragraph.text)
        return "\n".join(text_parts)
    except Exception as e:
        return f"Error extracting Word document: {str(e)}"

def _resource_to_context(resource) -> str:
    try:
        if isinstance(resource, (dict, list)):
            txt = json.dumps(resource, ensure_ascii=False, indent=2)
        else:
            txt = str(resource)
    except Exception:
        txt = str(resource)
    txt = txt.strip()
    if len(txt) > MAX_RESOURCE_CHARS:
        txt = txt[:MAX_RESOURCE_CHARS] + "\n…[truncated]"
    return txt

# =========================
# System (display only)
# =========================
SYSTEM_SIMPLE = """
You are Neuro Case Builder. Produce clear, medically plausible, and pedagogically aligned cases
for neurogenic communication disorders.

User-facing output order:
1) Case Narrative — weave PCC (identity, culture/language, roles, participation goals, supports, barriers, preferences) into HPI/PMH/ROS-communication/exam observations; person-first language.
2) Guiding Questions — ONLY if requested; 3–6 prompts for analysis.
3) Treatment Demonstration (RTSS) — ONLY if requested; short narrative + compact list: target (observable), ingredients, mechanism, dose/schedule, measure+cadence, success threshold.
4) Teaching Artifacts — ONLY if requested; language samples (error tags), test item responses, motor speech descriptors, short discourse transcript, simulated scores (clearly labeled).
"""

# =========================
# Agent prompts (JSON-only)
# =========================
CLARIFIER_PROMPT = """
You are the Objective Clarifier agent.
Purpose: Translate instructor goals into case specs (disorder focus, severity, clinical reasoning level, etc.).
Inputs: Free-text learning objectives, constraints.
Outputs: a JSON object with fields exactly:
{
  "disorders": [],
  "severity": "",
  "onset": "",
  "setting": "",
  "treatment_requested": false,
  "goals": [],
  "assumptions": [],
  "unresolved_questions": [],
  "audience_level": "novice|intermediate|advanced"
}
Instructions: Extract learning goals, disorders, severity, onset/setting, RTSS focus, audience_level (infer if missing).
Return ONLY a single JSON block (no prose) inside ```json fences.
"""

PROFILE_PROMPT = """
You are the Disorder Profile Designer.
Using the normalized spec and clinical references, propose a medically plausible profile:
- lesion/site + etiology, onset/course consistent with REF.clinical.onset_course
- hallmark signs consistent with REF.clinical.lesion_symptom
- common comorbidities
- differentials to rule out
Return ONLY JSON:
{
  "etiology": "",
  "lesion_site": [],
  "onset": "",
  "course": "",
  "hallmark_signs": [],
  "comorbidities": [],
  "differentials": []
}
"""

NARRATIVE_PROMPT = """
You are the Narrative & PCC Composer.
Write a coherent clinical narrative at the requested reading level.

Requirements:
- Create a flowing narrative paragraph (180-250 words) that naturally weaves together PCC and clinical information
- PCC elements to integrate: identity (name, age, pronouns, languages, cultural background), roles, participation goals, supports, barriers, preferences
- Clinical elements: HPI, PMH, ROS-communication, exam observations
- Use person-first language and include 1–2 realistic short quotes from the client and/or a familiar partner
- Tone: clinical yet respectful, appropriate to the reading level
- The narrative_text should be a single cohesive story, not separate sections

Return ONLY JSON:
{
  "pcc": { "identity": {}, "roles": [], "participation_goals": [], "supports": [], "barriers": [], "preferences": "" },
  "clinical_history": { "HPI":"", "PMH":"", "ros_communication":"", "exam_observations":"" },
  "narrative_text": "<THE FULL NARRATIVE PARAGRAPH HERE>"
}
"""

ARTIFACTS_PROMPT = """
You are the Teaching Artifact Synthesizer.
From the narrative (and spec/profile), produce succinct artifacts:
- language sample with inline error tags [phonemic], [semantic], [neologism], [perseveration], [agrammatism]
- test item responses (naming, repetition, comprehension) + brief scoring notes
- motor speech descriptors (DDK, prosody, voice, articulation, rate)
- short discourse transcript (picture description or procedural)
- simulated scores (CIU/min, PCC, SIT %) — clearly labeled 'simulated'
Return ONLY JSON: { "artifacts": { "language_sample": "", "test_items": [], "motor_speech": "", "discourse": "", "scores": {} } }
"""

RTSS_PLANNER_PROMPT = """
You are the RTSS Treatment Planner.
If treatment is requested, specify targets, ingredients, mechanisms, dose/schedule, measure+cadence, success thresholds; include 2–3 brief session plans and home practice. Align to evidence and REF.rtss.measure_registry.
Return ONLY JSON:
{
  "rtss": {
    "targets": [{
      "name": "", "construct": "",
      "ingredients": [{"name":"", "delivery":"", "materials":""}],
      "mechanism": "",
      "dose": {"session_min": null, "sessions_per_week": null, "total_weeks": null, "home_min_per_day": null},
      "schedule": "",
      "measures": [{"name":"", "cadence": [], "direction": "higher=better|lower=better", "success": ""}]
    }],
    "session_plans": [{"title":"","activities":[],"duration_min":0}]
  }
}
"""

AUDITOR_PROMPT = """
You are the Medical Plausibility & Bias Auditor.
Check for: lesion–symptom mismatches, implausible onset/course, unit/schedule issues, missing PCC, biased/stigmatizing phrasing. Use REF clinical and pedagogy data.
Return ONLY JSON:
{
  "issues": [{"type":"plausibility|bias|consistency","message":"", "suggested_fix": ""}],
  "severity": "ok|minor|moderate|critical"
}
"""

MI_GAS_PROMPT = """
You are the MI & GAS Coach.
Create:
1) Brief motivational interviewing dialogue (8–12 turns) between clinician (C) and client (Pt). Use MI spirit (collaboration, evocation, autonomy support). Label OARS moves (e.g., [Open Q], [Complex Reflection], [Affirmation], [Summary]).
2) A Goal Attainment Scale (GAS) for one priority participation goal. Provide:
   - One clear goal statement (observable, context-bound)
   - Anchors for -2, -1, 0, +1, +2 (0 = expected after intervention)
   - Baseline mapping (which level and why)
   - Measure + cadence and success threshold
Return ONLY JSON:
{
  "mi_dialogue": ["C: ... [Open Q]", "Pt: ...", "C: ... [Complex Reflection]"],
  "gas": {
    "goal": "",
    "anchors": {"-2":"","-1":"","0":"","+1":"","+2":""},
    "baseline_level": "-2|-1|0|+1|+2",
    "measurement": {"tool":"", "cadence": [], "success": ""}
  }
}
"""

CASE_EDITOR_PROMPT = """
You are the Case Editor agent.
Purpose: Revise an existing clinical case per user instructions while keeping it medically plausible and pedagogically aligned.

Inputs:
1) User edit request (free text)
2) Current case JSON (meta, pcc, clinical_history, rtss, artifacts, mi_gas if present)

Output: ONLY a single valid JSON object with the revised case (same top-level shape as input).

Editing guidelines:
- Preserve internal consistency: if etiology/lesion/setting change, align symptoms, narrative, and RTSS accordingly.
- Keep PCC tone and reading level consistent with the existing meta.reading_level.
- Only modify what is relevant to the edit request; do not delete unrelated sections.
- If you add new targets, include measures and success thresholds; use existing measure registry conventions when possible.
- ALWAYS update the narrative_text field in clinical_history to reflect any changes
Return ONLY a single JSON block (no prose) inside ```json fences.
"""

# =========================
# Utilities (parsing + call)
# =========================
JSON_FENCE_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def parse_fenced_json(text: str) -> Optional[Dict[str, Any]]:
    block = None
    for m in JSON_FENCE_RE.finditer(text or ""):
        block = m.group(1)
    if not block:
        try:
            return json.loads(text.strip())
        except Exception:
            return None
    try:
        return json.loads(block)
    except Exception:
        return None

def call_json_agent(system_prompt: str, user_payload: str, temperature: float = 0.3) -> Dict[str, Any]:
    """Call the model with streaming; collect full text while displaying live output."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload + "\n\nReturn ONLY one JSON object in ```json fences."}
    ]

    full_text = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream = client.chat.completions.create(
            model=MODEL_ID,
            temperature=temperature,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
                placeholder.markdown(full_text + "▌")

        placeholder.markdown(full_text)

    data = parse_fenced_json(full_text) or {}
    return data

def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

# =========================
# Agent wrappers
# =========================
def run_objective_clarifier(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text.strip():
        return None
    payload = f"OBJECTIVES/REQUEST:\n{raw_text}\n\nREF.pedagogy.learning_goal_map:\n{pretty(REF['pedagogy']['learning_goal_map'])}"
    data = call_json_agent(CLARIFIER_PROMPT, payload, temperature=0.2)
    return data or None

def run_profile_designer(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = f"SPEC:\n```json\n{pretty(spec)}\n```\nREF.clinical.lesion_symptom:\n{pretty(REF['clinical']['lesion_symptom'])}\nREF.clinical.onset_course:\n{pretty(REF['clinical']['onset_course'])}"
    data = call_json_agent(PROFILE_PROMPT, payload, temperature=0.2)
    return data or None

def run_narrative_pcc(spec: Dict[str, Any], profile: Dict[str, Any], reading_level: int) -> Optional[Dict[str, Any]]:
    level_text = "college level" if reading_level >= 14 else f"grade {reading_level}"
    payload = f"SPEC:\n```json\n{pretty(spec)}\n```\nPROFILE:\n```json\n{pretty(profile)}\n```\nICF EXAMPLES:\n{pretty(REF['pedagogy']['icf_examples'])}\nREADING LEVEL: {level_text}"
    data = call_json_agent(NARRATIVE_PROMPT, payload, temperature=0.5)
    return data or None

def run_artifacts(narrative: Dict[str, Any], spec: Dict[str, Any], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = f"NARRATIVE:\n```json\n{pretty(narrative)}\n```\nSPEC:\n```json\n{pretty(spec)}\n```\nPROFILE:\n```json\n{pretty(profile)}\n```\nTEST NORMS:\n{pretty(REF['clinical']['test_norms'])}"
    data = call_json_agent(ARTIFACTS_PROMPT, payload, temperature=0.4)
    return data or None

def run_rtss_planner(narrative: Dict[str, Any], spec: Dict[str, Any], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    extra = ""
    if "user_resource" in st.session_state:
        extra = "\nINSTRUCTOR RESOURCE:\n```\n" + _resource_to_context(st.session_state["user_resource"]) + "\n```"
    payload = (
        f"NARRATIVE:\n```json\n{pretty(narrative)}\n```\nSPEC:\n```json\n{pretty(spec)}\n```\nPROFILE:\n```json\n{pretty(profile)}\n```\n"
        f"REF.rtss.measure_registry:\n{pretty(REF['rtss']['measure_registry'])}\n"
        f"REF.rtss.protocols:\n{pretty(REF['rtss']['protocols'])}\n"
        f"REF.rtss.evidence (top 3 shown):\n{pretty(REF['rtss']['evidence'][:3])}"
        f"{extra}"
    )
    data = call_json_agent(RTSS_PLANNER_PROMPT, payload, temperature=0.4)
    return data or None

def run_auditor(full_text: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = (
        f"FULL TEXT (narrative + sections):\n{full_text[:8000]}\n\nMETA:\n```json\n{pretty(meta)}\n```\n"
        f"REF.clinical.lesion_symptom:\n{pretty(REF['clinical']['lesion_symptom'])}\n"
        f"REF.clinical.onset_course:\n{pretty(REF['clinical']['onset_course'])}\n"
        f"REF.pedagogy.reasoning_rubrics:\n{pretty(REF['pedagogy']['reasoning_rubrics'])}"
    )
    data = call_json_agent(AUDITOR_PROMPT, payload, temperature=0.0)
    return data or None

def run_mi_gas(narrative: Dict[str, Any], spec: Dict[str, Any], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = (
        f"NARRATIVE:\n```json\n{pretty(narrative)}\n```\n"
        f"SPEC:\n```json\n{pretty(spec)}\n```\n"
        f"PROFILE:\n```json\n{pretty(profile)}\n```\n"
        f"MI CHEATSHEET:\n{pretty(REF['pedagogy'].get('mi_cheatsheet', {}))}\n"
        f"GAS TEMPLATES:\n{pretty(REF['pedagogy'].get('gas_templates', {}))}\n"
    )
    data = call_json_agent(MI_GAS_PROMPT, payload, temperature=0.4)
    return data or None

def run_case_editor(edit_text: str, case_json: dict) -> Optional[Dict[str, Any]]:
    extra = ""
    if "user_resource" in st.session_state:
        extra = "\nINSTRUCTOR RESOURCE:\n```\n" + _resource_to_context(st.session_state["user_resource"]) + "\n```"
    payload = (
        f"EDIT REQUEST:\n{edit_text}\n\n"
        f"CURRENT CASE JSON:\n```json\n{pretty(case_json)}\n```\n"
        f"{extra}"
    )
    data = call_json_agent(CASE_EDITOR_PROMPT, payload, temperature=0.35)
    return data or None

def assemble_case(spec, narrative, rtss=None, artifacts=None, mi_gas=None) -> Dict[str, Any]:
    meta = {
        "disorders": spec.get("disorders", []),
        "severity": spec.get("severity", ""),
        "onset": spec.get("onset", ""),
        "setting": spec.get("setting", ""),
        "reading_level": spec.get("reading_level", "college"),
    }
    out = {
        "meta": meta,
        "pcc": narrative.get("pcc", {}),
        "clinical_history": narrative.get("clinical_history", {}),
        "narrative_text": narrative.get("narrative_text", "")
    }
    if rtss and rtss.get("rtss"):
        out["rtss"] = rtss["rtss"]
    if artifacts and artifacts.get("artifacts"):
        out["artifacts"] = artifacts["artifacts"]
    if mi_gas and (mi_gas.get("mi_dialogue") or mi_gas.get("gas")):
        out["mi_gas"] = mi_gas
    return out

# =========================
# Export functions
# =========================
def generate_formatted_text(case_json: dict) -> str:
    """Generate formatted plain text for Word/PDF export"""
    lines = []
    lines.append("NEUROGENIC COMMUNICATION CASE STUDY")
    lines.append("=" * 60)
    lines.append("")
    
    # Meta information
    meta = case_json.get("meta", {})
    lines.append(f"Disorder(s): {', '.join(meta.get('disorders', []))}")
    lines.append(f"Severity: {meta.get('severity', 'N/A')}")
    lines.append(f"Onset: {meta.get('onset', 'N/A')}")
    lines.append(f"Setting: {meta.get('setting', 'N/A')}")
    lines.append("")
    
    # Narrative
    if case_json.get("narrative_text"):
        lines.append("CASE NARRATIVE")
        lines.append("-" * 60)
        lines.append(case_json["narrative_text"])
        lines.append("")
    
    # PCC Summary
    pcc = case_json.get("pcc", {})
    if pcc:
        lines.append("PERSON-CENTERED CARE SUMMARY")
        lines.append("-" * 60)
        identity = pcc.get("identity", {})
        if identity:
            lines.append(f"Identity: {json.dumps(identity)}")
        if pcc.get("roles"):
            lines.append(f"Roles: {', '.join(pcc['roles'])}")
        if pcc.get("participation_goals"):
            lines.append(f"Goals: {', '.join(pcc['participation_goals'])}")
        if pcc.get("supports"):
            lines.append(f"Supports: {', '.join(pcc['supports'])}")
        if pcc.get("barriers"):
            lines.append(f"Barriers: {', '.join(pcc['barriers'])}")
        if pcc.get("preferences"):
            lines.append(f"Preferences: {pcc['preferences']}")
        lines.append("")
    
    # Clinical History
    chx = case_json.get("clinical_history", {})
    if chx:
        lines.append("CLINICAL HISTORY")
        lines.append("-" * 60)
        for key in ["HPI", "PMH", "ros_communication", "exam_observations"]:
            if chx.get(key):
                lines.append(f"\n{key.upper()}:")
                lines.append(chx[key])
        lines.append("")
    
    # RTSS Treatment
    rtss = case_json.get("rtss")
    if rtss:
        lines.append("TREATMENT PLAN (RTSS)")
        lines.append("-" * 60)
        lines.append(json.dumps(rtss, indent=2))
        lines.append("")
    
    # Artifacts
    artifacts = case_json.get("artifacts")
    if artifacts:
        lines.append("TEACHING ARTIFACTS")
        lines.append("-" * 60)
        lines.append(json.dumps(artifacts, indent=2))
        lines.append("")
    
    # MI & GAS
    mi_gas = case_json.get("mi_gas")
    if mi_gas:
        lines.append("MOTIVATIONAL INTERVIEWING & GOAL ATTAINMENT SCALING")
        lines.append("-" * 60)
        lines.append(json.dumps(mi_gas, indent=2))
        lines.append("")
    
    return "\n".join(lines)

# =========================
# Sidebar: instructor upload
# =========================
with st.sidebar:
    st.header("📤 Upload resource")
    uploaded_file = st.file_uploader(
        "JSON or TXT (used as context for the next generation)",
        type=["json", "txt"], accept_multiple_files=False
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/json" or uploaded_file.name.lower().endswith(".json"):
                st.session_state["user_resource"] = json.load(uploaded_file)
                preview = _preview_text(json.dumps(st.session_state["user_resource"], ensure_ascii=False))
                st.success(f"Loaded JSON: {uploaded_file.name}")
                st.caption(f"Preview:\n{preview}")
            else:
                text = uploaded_file.read().decode(errors="ignore")
                st.session_state["user_resource"] = text
                st.success(f"Loaded text: {uploaded_file.name}")
                st.caption(f"Preview:\n{_preview_text(text)}")
        except Exception as e:
            st.error(f"Upload error: {e}")

    if "user_resource" in st.session_state:
        if st.button("Clear uploaded resource", use_container_width=True):
            del st.session_state["user_resource"]
            st.experimental_rerun()

# =========================
# Options
# =========================
with st.expander("Options", expanded=False):
    show_questions = st.checkbox("Include guiding questions", value=False)
    include_tx = st.checkbox("Include treatment plan (RTSS)", value=True)
    include_artifacts = st.checkbox("Include teaching artifacts", value=False)
    include_mi_gas = st.checkbox("Include MI dialogue + GAS goal", value=False)
    reading_level = st.select_slider(
        "Target reading level",
        options=list(range(6, 15)),
        value=14,
        format_func=lambda x: "College" if x >= 14 else f"Grade {x}",
    )
    show_export_options = st.checkbox("Show export options", value=True)

# =========================
# Session state
# =========================
class Message(TypedDict):
    role: str
    content: str

if "messages" not in st.session_state:
    msgs: List[Message] = [
        {"role": "system", "content": SYSTEM_SIMPLE},
        {"role": "assistant", "content": "Hi! What case would you like me to generate?"}
    ]
    st.session_state.messages = msgs

if "last_case_json" not in st.session_state:
    st.session_state.last_case_json = None
if "clarified_spec" not in st.session_state:
    st.session_state.clarified_spec = None
if "last_profile" not in st.session_state:
    st.session_state.last_profile = None
if "last_narrative" not in st.session_state:
    st.session_state.last_narrative = None

# =========================
# Render chat history
# =========================
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Chat input & commands
# =========================
user_input = st.chat_input(
    "Type a request. Tips: /clarify …, /tx, /artifacts, /mi, /audit, /edit your change"
)

def render_and_store_assistant(text: str):
    st.session_state.messages.append({"role": "assistant", "content": text})
    st.experimental_rerun()

if user_input:
    stripped = user_input.strip()

    # ----- /clarify -----
    if stripped.lower().startswith("/clarify"):
        payload = stripped[len("/clarify"):].strip()
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Clarifying objectives..."):
            spec = run_objective_clarifier(payload)
            if spec:
                spec["reading_level"] = "college"
                st.session_state.clarified_spec = spec
                render_and_store_assistant(f"**Objective Clarifier**\n```json\n{pretty(spec)}\n```")
            else:
                render_and_store_assistant("⚠️ No valid spec JSON returned.")

    # ----- /tx -----
    elif stripped.lower().startswith("/tx"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating RTSS treatment plan..."):
            if not st.session_state.last_narrative or not st.session_state.clarified_spec or not st.session_state.last_profile:
                render_and_store_assistant("⚠️ Generate a case first (so I have narrative/spec/profile), then use /tx.")
            else:
                rtss = run_rtss_planner(st.session_state.last_narrative, st.session_state.clarified_spec, st.session_state.last_profile)
                render_and_store_assistant(f"**RTSS Plan**\n```json\n{pretty(rtss)}\n```")

    # ----- /artifacts -----
    elif stripped.lower().startswith("/artifacts"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating teaching artifacts..."):
            if not st.session_state.last_narrative or not st.session_state.clarified_spec or not st.session_state.last_profile:
                render_and_store_assistant("⚠️ Generate a case first, then use /artifacts.")
            else:
                art = run_artifacts(st.session_state.last_narrative, st.session_state.clarified_spec, st.session_state.last_profile)
                render_and_store_assistant(f"**Teaching Artifacts**\n```json\n{pretty(art)}\n```")

    # ----- /mi -----
    elif stripped.lower().startswith("/mi"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating MI dialogue + GAS goal..."):
            if not st.session_state.last_narrative or not st.session_state.clarified_spec or not st.session_state.last_profile:
                render_and_store_assistant("⚠️ Generate a case first, then use /mi.")
            else:
                migas = run_mi_gas(st.session_state.last_narrative, st.session_state.clarified_spec, st.session_state.last_profile)
                render_and_store_assistant(f"**MI + GAS**\n```json\n{pretty(migas)}\n```")

    # ----- /audit -----
    elif stripped.lower().startswith("/audit"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Auditing last output..."):
            if not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant":
                render_and_store_assistant("⚠️ No assistant output to audit.")
            else:
                last_text = st.session_state.messages[-1]["content"]
                meta = (st.session_state.last_case_json or {}).get("meta", {})
                audit = run_auditor(last_text, meta)
                render_and_store_assistant(f"**Audit Report**\n```json\n{pretty(audit)}\n```")

    # ----- /edit -----
    elif stripped.lower().startswith("/edit"):
        edit_text = stripped[len("/edit"):].strip()
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Editing current case..."):
            if not st.session_state.get("last_case_json"):
                render_and_store_assistant("⚠️ No existing case to edit. Generate one first, then use `/edit your change`.")
            else:
                revised = run_case_editor(edit_text, st.session_state["last_case_json"])
                if not revised:
                    render_and_store_assistant("⚠️ Edit failed — no valid revised JSON returned. Try being more specific.")
                else:
                    st.session_state["last_case_json"] = revised
                    meta = revised.get("meta", {})
                    
                    md = []
                    md.append("### Revised Case Narrative")
                    
                    # Display the narrative text
                    if revised.get("narrative_text"):
                        md.append(revised["narrative_text"])
                    
                    md.append(f"\n**Meta**: {', '.join(meta.get('disorders', []))} | {meta.get('severity','')} | {meta.get('onset','')} | {meta.get('setting','')}")
                    
                    rtss = revised.get("rtss")
                    art = revised.get("artifacts")
                    mig = revised.get("mi_gas")

                    if rtss:
                        md.append("\n### Treatment (RTSS)")
                        md.append("```json\n" + pretty(rtss) + "\n```")
                    if art:
                        md.append("\n### Teaching Artifacts")
                        md.append("```json\n" + pretty(art) + "\n```")
                    if mig:
                        md.append("\n### MI Dialogue & GAS")
                        md.append("```json\n" + pretty(mig) + "\n```")

                    audit = run_auditor("\n".join(md), meta)
                    if audit and audit.get("issues"):
                        md.append("\n### Audit Report")
                        md.append("```json\n" + pretty(audit) + "\n```")

                    md.append("\n### Export JSON (Revised)\n```json\n" + pretty(revised) + "\n```")
                    render_and_store_assistant("\n".join(md))

    # ----- Check if user wants to edit existing case -----
    elif st.session_state.get("last_case_json") and any(keyword in stripped.lower() for keyword in ["edit", "change", "modify", "update", "revise", "adjust", "fix"]):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Editing current case..."):
            revised = run_case_editor(stripped, st.session_state["last_case_json"])
            if not revised:
                render_and_store_assistant("⚠️ Edit failed — no valid revised JSON returned. Try being more specific.")
            else:
                st.session_state["last_case_json"] = revised
                meta = revised.get("meta", {})
                
                md = []
                md.append("### Revised Case Narrative")
                
                # Display the narrative text
                if revised.get("narrative_text"):
                    md.append(revised["narrative_text"])
                
                md.append(f"\n**Meta**: {', '.join(meta.get('disorders', []))} | {meta.get('severity','')} | {meta.get('onset','')} | {meta.get('setting','')}")
                
                rtss = revised.get("rtss")
                art = revised.get("artifacts")
                mig = revised.get("mi_gas")

                if rtss:
                    md.append("\n### Treatment (RTSS)")
                    # Format RTSS as readable text
                    if rtss.get("targets"):
                        for target in rtss["targets"]:
                            md.append(f"\n**Target**: {target.get('name', 'N/A')}")
                            md.append(f"**Construct**: {target.get('construct', 'N/A')}")
                            md.append(f"**Mechanism**: {target.get('mechanism', 'N/A')}")
                            
                            if target.get("ingredients"):
                                md.append("\n**Ingredients**:")
                                for ing in target["ingredients"]:
                                    md.append(f"- {ing.get('name', 'N/A')}: {ing.get('delivery', 'N/A')} | Materials: {ing.get('materials', 'N/A')}")
                            
                            dose = target.get("dose", {})
                            md.append(f"\n**Dose**: {dose.get('session_min', 'N/A')} min/session, {dose.get('sessions_per_week', 'N/A')}x/week for {dose.get('total_weeks', 'N/A')} weeks")
                            if dose.get('home_min_per_day'):
                                md.append(f"**Home practice**: {dose['home_min_per_day']} min/day")
                
                if art:
                    md.append("\n### Teaching Artifacts")
                    if art.get("language_sample"):
                        md.append(f"\n**Language Sample**:\n{art['language_sample']}")
                    if art.get("motor_speech"):
                        md.append(f"\n**Motor Speech**:\n{art['motor_speech']}")
                    if art.get("discourse"):
                        md.append(f"\n**Discourse**:\n{art['discourse']}")
                
                if mig:
                    md.append("\n### MI Dialogue & GAS")
                    if mig.get("mi_dialogue"):
                        md.append("\n**Dialogue**:")
                        for turn in mig["mi_dialogue"]:
                            md.append(turn)
                    if mig.get("gas"):
                        md.append(f"\n**GAS Goal**: {mig['gas'].get('goal', 'N/A')}")

                audit = run_auditor("\n".join(md), meta)
                if audit and audit.get("issues"):
                    md.append("\n### Audit Report")
                    for issue in audit["issues"]:
                        md.append(f"- **{issue.get('type', 'issue').upper()}**: {issue.get('message', 'N/A')}")

                render_and_store_assistant("\n".join(md))

    # ----- Check if user wants to edit existing case -----
    elif st.session_state.get("last_case_json") and any(keyword in stripped.lower() for keyword in ["edit", "change", "modify", "update", "revise", "adjust", "fix"]):
        directives = (
            f"\n\nOptions:\n"
            f"- Include guiding questions: {show_questions}\n"
            f"- Include treatment plan (RTSS): {include_tx}\n"
            f"- Include teaching artifacts: {include_artifacts}\n"
            f"- Include MI dialogue + GAS: {include_mi_gas}\n"
            f"- Reading level: {'college' if reading_level >= 14 else f'grade {reading_level}'}\n"
        )
        st.session_state.messages.append({"role": "user", "content": user_input + directives})

        with st.spinner("Building case (clarify → profile → narrative → optional RTSS/artifacts/MI+GAS → audit)..."):
            # 1) Clarify
            spec = run_objective_clarifier(user_input) or {
                "disorders": [], "severity": "", "onset": "", "setting": "", "treatment_requested": include_tx,
                "goals": [], "assumptions": ["defaults used"], "unresolved_questions": [], "audience_level": "advanced"
            }
            spec["reading_level"] = "college" if reading_level >= 14 else f"grade {reading_level}"
            st.session_state.clarified_spec = spec

            # 2) Profile
            profile = run_profile_designer(spec) or {
                "etiology": "stroke", "lesion_site": ["left IFG"], "onset": spec.get("onset","subacute"),
                "course": "improving", "hallmark_signs": [], "comorbidities": [], "differentials": []
            }
            st.session_state.last_profile = profile

            # 3) Narrative & PCC
            narrative = run_narrative_pcc(spec, profile, reading_level) or {"pcc": {}, "clinical_history": {}, "narrative_text": ""}
            st.session_state.last_narrative = narrative

            # 4) Optional: RTSS + Artifacts + MI/GAS
            rtss = run_rtss_planner(narrative, spec, profile) if (include_tx or spec.get("treatment_requested")) else None
            artifacts = run_artifacts(narrative, spec, profile) if include_artifacts else None
            mi_gas = run_mi_gas(narrative, spec, profile) if include_mi_gas else None

            # 5) Assemble case JSON
            case_json = assemble_case(spec, narrative, rtss, artifacts, mi_gas)
            st.session_state.last_case_json = case_json

            # 6) Build narrative-first display
            narrative_md = []
            narrative_md.append("### Case Narrative")
            
            # Display the actual narrative text
            if narrative.get("narrative_text"):
                narrative_md.append(narrative["narrative_text"])
            
            # Add metadata
            meta = case_json.get("meta", {})
            narrative_md.append(f"\n**Meta**: {', '.join(meta.get('disorders', []))} | {meta.get('severity','')} | {meta.get('onset','')} | {meta.get('setting','')}")

            if show_questions:
                narrative_md.append("\n### Guiding Questions\n- What are the priority participation goals?\n- Which impairments limit activity the most?\n- What measure will you use to track progress?\n- Which mechanism justifies your chosen ingredients?")

            if rtss and rtss.get("rtss"):
                narrative_md.append("\n### Treatment Demonstration (RTSS)")
                # Format RTSS as readable text instead of JSON
                rtss_data = rtss["rtss"]
                if rtss_data.get("targets"):
                    for target in rtss_data["targets"]:
                        narrative_md.append(f"\n**Target**: {target.get('name', 'N/A')}")
                        narrative_md.append(f"**Construct**: {target.get('construct', 'N/A')}")
                        narrative_md.append(f"**Mechanism**: {target.get('mechanism', 'N/A')}")
                        
                        if target.get("ingredients"):
                            narrative_md.append("\n**Ingredients**:")
                            for ing in target["ingredients"]:
                                narrative_md.append(f"- {ing.get('name', 'N/A')}: {ing.get('delivery', 'N/A')} | Materials: {ing.get('materials', 'N/A')}")
                        
                        dose = target.get("dose", {})
                        narrative_md.append(f"\n**Dose**: {dose.get('session_min', 'N/A')} min/session, {dose.get('sessions_per_week', 'N/A')}x/week for {dose.get('total_weeks', 'N/A')} weeks")
                        if dose.get('home_min_per_day'):
                            narrative_md.append(f"**Home practice**: {dose['home_min_per_day']} min/day")
                        
                        if target.get("measures"):
                            narrative_md.append("\n**Measures**:")
                            for measure in target["measures"]:
                                narrative_md.append(f"- {measure.get('name', 'N/A')} ({measure.get('direction', 'N/A')}): {', '.join(measure.get('cadence', []))} | Success: {measure.get('success', 'N/A')}")
                        narrative_md.append("")
                
                if rtss_data.get("session_plans"):
                    narrative_md.append("**Session Plans**:")
                    for plan in rtss_data["session_plans"]:
                        narrative_md.append(f"\n*{plan.get('title', 'Session')}* ({plan.get('duration_min', 'N/A')} min)")
                        if plan.get("activities"):
                            for activity in plan["activities"]:
                                narrative_md.append(f"  - {activity}")

            if artifacts and artifacts.get("artifacts"):
                narrative_md.append("\n### Teaching Artifacts")
                art_data = artifacts["artifacts"]
                
                if art_data.get("language_sample"):
                    narrative_md.append(f"\n**Language Sample**:\n{art_data['language_sample']}")
                
                if art_data.get("test_items"):
                    narrative_md.append("\n**Test Item Responses**:")
                    for item in art_data["test_items"]:
                        if isinstance(item, dict):
                            narrative_md.append(f"- {item}")
                        else:
                            narrative_md.append(f"- {item}")
                
                if art_data.get("motor_speech"):
                    narrative_md.append(f"\n**Motor Speech Observations**:\n{art_data['motor_speech']}")
                
                if art_data.get("discourse"):
                    narrative_md.append(f"\n**Discourse Transcript**:\n{art_data['discourse']}")
                
                if art_data.get("scores"):
                    narrative_md.append("\n**Simulated Assessment Scores**:")
                    for key, val in art_data["scores"].items():
                        narrative_md.append(f"- {key}: {val}")

            if mi_gas and (mi_gas.get("mi_dialogue") or mi_gas.get("gas")):
                narrative_md.append("\n### Motivational Interviewing Dialogue & Goal Attainment Scaling")
                
                if mi_gas.get("mi_dialogue"):
                    narrative_md.append("\n**MI Dialogue**:")
                    for turn in mi_gas["mi_dialogue"]:
                        narrative_md.append(turn)
                
                if mi_gas.get("gas"):
                    gas_data = mi_gas["gas"]
                    narrative_md.append(f"\n**GAS Goal**: {gas_data.get('goal', 'N/A')}")
                    narrative_md.append(f"\n**Baseline Level**: {gas_data.get('baseline_level', 'N/A')}")
                    
                    if gas_data.get("anchors"):
                        narrative_md.append("\n**Scale Anchors**:")
                        for level in ["-2", "-1", "0", "+1", "+2"]:
                            if level in gas_data["anchors"]:
                                narrative_md.append(f"- **{level}**: {gas_data['anchors'][level]}")
                    
                    if gas_data.get("measurement"):
                        meas = gas_data["measurement"]
                        narrative_md.append(f"\n**Measurement**: {meas.get('tool', 'N/A')} | Cadence: {', '.join(meas.get('cadence', []))} | Success: {meas.get('success', 'N/A')}")

            # 7) Audit
            audit = run_auditor("\n".join(narrative_md), case_json.get("meta", {}))
            if audit and audit.get("issues"):
                narrative_md.append("\n### Audit Report")
                for issue in audit["issues"]:
                    narrative_md.append(f"- **{issue.get('type', 'issue').upper()}**: {issue.get('message', 'N/A')}")
                    if issue.get("suggested_fix"):
                        narrative_md.append(f"  - *Suggested fix*: {issue['suggested_fix']}")
                narrative_md.append(f"\n**Severity**: {audit.get('severity', 'unknown')}")

            # 8) Don't append JSON export anymore

            render_and_store_assistant("\n".join(narrative_md))

# ================
# Export buttons (conditional)
# ================
if show_export_options:
    def extract_json_from_last_assistant() -> Optional[Dict[str, Any]]:
        fence = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
        for m in reversed(st.session_state.messages):
            if m.get("role") == "assistant":
                text = m.get("content","")
                blocks = fence.findall(text)
                if blocks:
                    try:
                        return json.loads(blocks[-1])
                    except Exception:
                        continue
        return None

    export_data = extract_json_from_last_assistant()
    if export_data:
        # Plain text export only (can be opened in Word)
        formatted_text = generate_formatted_text(export_data)
        st.download_button(
            "⬇️ Download TXT (Word-compatible)",
            data=formatted_text,
            file_name=f"neuro_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )