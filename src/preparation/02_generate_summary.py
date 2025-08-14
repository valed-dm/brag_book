from typing import Any
from typing import Dict

import pandas as pd

from src.ai_engine.generator import LLMGenerator
from src.core.config import settings
from src.core.logging_setup import log


def analyze_case_studies(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes the dataframe to extract key statistics for the summary prompt."""
    category_counts: Dict[str, int] = df.iloc[:, 2].value_counts().to_dict()

    titles: list[str] = df.iloc[:, 1].tolist()
    examples = [
        title
        for title in titles
        if any(
            keyword in title.lower()
            for keyword in ["lamoda", "s7", "газпромбанк", "leroy merlin", "dodo pizza"]
        )
    ]
    unique_examples: list[str] = list(set(examples))

    analysis: Dict[str, Any] = {
        "project_count": len(df),
        "category_counts": category_counts,
        "project_examples": unique_examples[:5],
    }
    log.info(f"Analysis complete: {analysis}")
    return analysis


def create_summary_prompt(analysis: Dict[str, Any]) -> str:
    """Creates a detailed prompt for the LLM to generate a company summary."""
    prompt = f"""
    You are EORA's lead brand strategist and storyteller.
    Your mission is to craft a compelling narrative that showcases our identity,
    expertise, and impact. Your primary task is to **synthesize**, not summarize.

    ## Audience Profile & Core Requirements:
    - **Audience**: Discerning executives seeking a strategic partner.
    - **Tone**: Confident, professional, and visionary.
    - **Output**: A polished, CEO-ready narrative of 2-3 powerful paragraphs.

    ## Crucial Style Guideline: Show, Don't Just Tell
    The data below is for **context and inspiration, not for direct quotation**.
    Weave it into a compelling story.

    **For example:**

    - **AVOID THIS (Listing Data):** "We completed 65 projects, with our top categories
    being voice (17) and chatbot (16)."

    - **DO THIS (Synthesizing Data):** "With a proven track record of over 60 successful
    engagements,  we have established ourselves as leaders in the conversational
    AI space, demonstrating deep expertise in transforming how businesses interact
    with their customers."

    ## Data Insights (for your context):
    - **Portfolio Size**: {analysis["project_count"]} successful engagements.
    - **Industry Footprint**: Our expertise is broad, touching on
    {list(analysis["category_counts"].keys())}, with deep specialization in areas
     like voice, conversational AI, and computer vision.
    - **Representative Clients**: Our portfolio includes collaborations with
    industry leaders such as {analysis["project_examples"]}.

    ## Narrative Framework:
    1.  **Strategic Hook**: Start with a powerful opening statement that defines
    EORA's unique value proposition.
    2.  **Evidence of Excellence**: Illustrate our expertise with the caliber of
    companies that trust us and our proven success across diverse industries.
    3.  **Visionary Close**: Conclude with a forward-looking statement about our
    role as a strategic transformation partner.

    Now, following these guidelines, craft the narrative.
    """

    return prompt


def clean_llm_output(raw_text: str) -> str:
    """
    Cleans up common artifacts from LLM-generated text, like conversational
    preambles and structural markdown labels.
    """
    if not raw_text:
        return ""

    # 1. Remove conversational intros (find the first real paragraph)
    # This regex looks for the first paragraph that doesn't start with
    # conversational phrases.
    # A simpler method is to split by newline and find the first non-empty line
    # that starts with a capital letter and is not a markdown header.
    lines = raw_text.strip().split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip empty lines and structural labels
        if not line.strip() or line.strip().startswith("**"):
            continue
        # A simple way to skip the preamble
        if "my attempt" in line.lower() or "here's the" in line.lower():
            continue
        cleaned_lines.append(line.strip())

    # Re-join the cleaned lines
    cleaned_text = "\n\n".join(cleaned_lines)  # Use double newline for paragraphs

    # A more aggressive removal of any remaining labels
    cleaned_text = cleaned_text.replace("**Strategic Hook**", "")
    cleaned_text = cleaned_text.replace("**Evidence of Excellence**", "")
    cleaned_text = cleaned_text.replace("**Visionary Close**", "")

    return cleaned_text.strip()


def main() -> None:
    """Main function to generate and save the company summary."""
    log.info("Starting summary generation process...")
    settings.KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
    summary_text = ""

    try:
        df = pd.read_csv(settings.INPUT_CSV_PATH, header=None, skiprows=[0])
        log.info(f"Successfully loaded {len(df)} records from CSV for analysis.")
    except FileNotFoundError:
        log.error(f"Input file not found at {settings.INPUT_CSV_PATH}.")
        return

    analysis_results = analyze_case_studies(df)
    summary_prompt = create_summary_prompt(analysis_results)
    log.info("Generated prompt for LLM.")

    try:
        log.info("Initializing LLM Generator...")
        generator = LLMGenerator()
        raw_summary_text = generator.generate(summary_prompt)
        if raw_summary_text:
            summary_text = clean_llm_output(raw_summary_text)
            log.info("Successfully received response from LLM.")

    except Exception as e:
        log.error(f"An error occurred during summary generation: {e}")
        return

    if summary_text:
        log.info(f"Saving generated summary to {settings.SUMMARY_PATH}")
        with open(settings.SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(summary_text)
        log.info("Company summary successfully saved.")
    else:
        log.error("Failed to generate summary: The LLM returned no content.")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
