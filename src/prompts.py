"""
prompts.py
~~~~~~~~~~
Builds the system prompt sent to the LLM for main case generation.
Keeping prompt logic here prevents main.py from becoming a wall of text.
"""

from config import DIFFICULTY_MAP


def build_system_prompt(
    *,
    difficulty: str,
    is_initial_assessment: bool,
    is_transfer_initial: bool,
    reference_content: str,
    ethnicity: str,
) -> str:
    """
    Construct the full system prompt for case generation.

    Parameters
    ----------
    difficulty            : One of the keys in DIFFICULTY_MAP.
    is_initial_assessment : True when the user asked for an initial assessment.
    is_transfer_initial   : True when it's an initial assessment during care transfer.
    reference_content     : Raw text of the best-matching exemplar (may be empty).
    ethnicity             : Randomly selected patient ethnicity string.
    """
    difficulty_label = DIFFICULTY_MAP.get(difficulty, "intermediate")

    # --- Reference style guide block ---
    reference_instruction = ""
    if reference_content:
        reference_instruction = (
            "\n### Reference Style Guide\n"
            "Below is an example of an ideal case study. Use it ONLY for structural and "
            "stylistic inspiration (tone, depth of detail, phrasing). "
            "DO NOT copy the patient details, diagnosis, or specific scenario from this "
            "reference. Apply this high-quality clinical writing style to the specific "
            "case requested by the user.\n\n"
            f"<reference_exemplar>\n{reference_content}\n</reference_exemplar>\n"
        )

    # --- Ethnicity instruction ---
    ethnicity_instruction = (
        f"\n\nIMPORTANT: For this case, make the patient {ethnicity}. Incorporate "
        "culturally appropriate details naturally into the case narrative (name, family "
        "dynamics, cultural considerations for treatment, etc.) without stereotyping."
    )

    # --- Assessment context instructions ---
    guiding_focus = (
        "Focus these questions on determining initial assessment priorities and "
        "selecting appropriate standardized and informal tools."
        if is_initial_assessment
        else "Use these questions to spark discussion about differential diagnosis, "
        "intervention planning, and clinical reasoning."
    )

    language_sample_note = (
        "Ensure the language sample reflects spontaneous speech observable prior to "
        "formal testing."
        if (is_initial_assessment and not is_transfer_initial)
        else ""
    )

    initial_assessment_guidance = ""
    if is_initial_assessment:
        if is_transfer_initial:
            initial_assessment_guidance = (
                "\nWhen the user signals an initial assessment during a transfer of care, "
                "summarize what prior SLPs accomplished, the patient's recovery trajectory, "
                "and documentation the new setting would inherit. Highlight unanswered "
                "assessment questions for the new site and clarify what still requires "
                "standardized and informal testing."
            )
        else:
            initial_assessment_guidance = (
                "\nWhen the user signals an initial assessment for a new onset with no prior "
                "SLP involvement, limit the case to referral information, medical history, "
                "observable behaviors, and collateral reports available before assessments "
                "are administered. Do not invent completed SLP assessment results yet; "
                "instead, describe what needs to be investigated and why."
            )

    return f"""You are an expert clinical educator specialising in speech-language pathology. \
Your role is to help university instructors create high-quality, realistic case studies for their students.
{reference_instruction}{ethnicity_instruction}
Always craft a single cohesive narrative case study (paragraph style, not bullet points) that weaves together \
(and do not pre-answer or give instructions for the guiding questions here):
- Patient demographics and background
- Detailed medical history and etiology
- Presenting symptoms and characteristics
- Assessment results (formal and informal) with interpretation
- Clinical observations and differential diagnosis considerations
- Treatment recommendations, prognosis.

Keep the narrative observational and exploratory; reserve any prioritisation, tool selection, or explicit answers \
for the Guiding Questions section only.

After the narrative, append two explicit sections. Bold the headers exactly as shown \
("**Language Sample**" and "**Guiding Questions**") so they stand out visually:
1. **Language Sample** — provide a short quoted transcript (4–6 sentences) that captures the client's spontaneous \
speech. Match the cadence of this reference format: "Well the, uh, the little… the little cookers are spinning up \
there and she's trying to wash the plates but the water's all, all floofing out. And the boy, he's, he's grabbing \
the stool cause he wants the cookie. They're having a good time, I think, and the mother doesn't know the window \
is, is, uh, smiling. It's pretty noisy in that room." Use it only as stylistic guidance; compose a fresh sample \
aligned with the case's disorder-specific features every time so it reflects the symptoms described in the user \
input. {language_sample_note}
2. **Guiding Questions** — provide discussion questions instructors can use in class. {guiding_focus}

Keep the entire response focused on the case (no general tips). Adjust complexity to {difficulty_label} level, \
ensure clinical accuracy, and use professional terminology appropriate for graduate-level speech-language pathology \
education.{initial_assessment_guidance}"""