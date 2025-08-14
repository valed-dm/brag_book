import argparse
import textwrap
from typing import Optional

# Import the master controller of our AI engine
from src.ai_engine.orchestrator import Orchestrator
from src.core.logging_setup import log


def main() -> None:
    """The main entry point for the Command-Line Interface."""

    parser = argparse.ArgumentParser(
        description="AI assistant for EORA."
        " Ask a question about the company's projects."
    )
    parser.add_argument("query", type=str, help="Your question in quotes.")
    parser.add_argument(
        "--lang",
        type=str,
        default="english",
        help="The desired output language"
        " (e.g., 'russian', 'english'). Defaults to english.",
    )

    args = parser.parse_args()
    user_query = args.query
    language_name = args.lang.lower()

    # --- Map full language names to standard 2-letter codes ---
    lang_code_map = {
        "russian": "ru",
        "english": "en",
        # Add other languages here if needed
    }
    # Default to 'en' if the language name isn't in our map
    language_code = lang_code_map.get(language_name, "en")

    if not user_query:
        print("Please provide a question.")
        return

    try:
        print("Initializing AI assistant... (This might take a moment)")
        orchestrator = Orchestrator()

        print("Thinking...")
        # Pass the correct language code to the orchestrator
        answer: Optional[str] = orchestrator.answer(user_query, output_language=language_code)

        if answer:
            print("\n" + "=" * 80)
            print("Ответ EORA ассистента:\n")
            print(textwrap.fill(answer, width=80))
            print("=" * 80 + "\n")
        else:
            print("\n--- EORA Assistant ---")
            print(
                "Sorry, an unexpected error occurred and I could not provide an answer."
            )

    except Exception as e:
        log.error(f"An error occurred in the application: {e}")
        print(
            "\nSorry, an unexpected error occurred."
            " Please check the logs for more details."
        )


if __name__ == "__main__":
    main()
