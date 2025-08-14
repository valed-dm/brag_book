from typing import Any
from typing import Union

import ollama
from openai import OpenAI

from src.core.config import settings
from src.core.logging_setup import log


class LLMGenerator:
    """A wrapper class for generating text using different LLM providers
    (Ollama or OpenAI).
    """

    provider: str
    client: Union[ollama.Client, OpenAI]

    def __init__(self) -> None:
        """Initializes the generator based on the configuration."""
        self.provider = settings.AI_PROVIDER.lower()
        log.info(f"Initializing LLMGenerator with provider: {self.provider}")

        if self.provider == "ollama":
            try:
                self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
                # Check connection
                self.client.list()
            except Exception as e:
                log.error(
                    f"Failed to connect to Ollama at {settings.OLLAMA_BASE_URL}. "
                    f"Please ensure the Ollama application is running. Error: {e}"
                )
                raise
        elif self.provider == "openai":
            self.client = OpenAI()
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")

    def generate(self, prompt: str) -> str | None | Any:
        """
        Generates text based on a prompt using the configured provider.

        Args:
            prompt: A string prompt for the LLM.

        Returns:
            The generated text as a string.
        """
        log.info(f"Generating text with {self.provider}...")

        messages = [{"role": "user", "content": prompt}]

        try:
            if self.provider == "ollama":
                ollama_response = ollama.chat(
                    model=settings.OLLAMA_MODEL_NAME,
                    messages=messages,
                )
                message = ollama_response.get("message", {})
                return message.get("content")

            elif self.provider == "openai":
                if isinstance(self.client, OpenAI):
                    openai_response = self.client.chat.completions.create(
                        model=settings.GENERATOR_MODEL_NAME,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=0.5,
                    )
                    if openai_response.choices:
                        return openai_response.choices[0].message.content
                    return None  # Return None if choices are empty
                else:
                    raise TypeError("Mismatched provider and client type.")

        except Exception as e:
            log.exception(
                f"An error occurred during LLM generation with {self.provider}: {e}"
            )
            raise

        return None

    def translate(self, text: str, target_language: str) -> str | None | Any:
        """
        Translates a given text to the target language using the configured LLM.

        Args:
            text: The text to translate.
            target_language: The language to translate to (e.g., "Russian", "English").

        Returns:
            The translated text.
        """
        log.info(f"Translating text to {target_language}...")

        prompt = (
            f"Translate the following text to {target_language}."
            f" Respond with only the translated text and nothing else:\n\n{text}"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            if self.provider == "ollama":
                ollama_response = ollama.chat(
                    model=settings.OLLAMA_MODEL_NAME,
                    messages=messages,
                )
                message = ollama_response.get("message", {})
                return message.get("content")

            elif self.provider == "openai":
                if isinstance(self.client, OpenAI):
                    openai_response = self.client.chat.completions.create(
                        model=settings.GENERATOR_MODEL_NAME,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=0.1,
                    )
                    if openai_response.choices:
                        return openai_response.choices[0].message.content
                    return None  # No choices returned
                else:
                    # This runtime check ensures logical consistency
                    raise TypeError("Mismatched provider and client type.")

        except Exception as e:
            log.error(f"An error occurred during translation: {e}")
            # Return None on failure instead of the original text
            return None

        # This handles the case where self.provider is not a recognized value.
        log.warning(
            f"Unsupported provider '{self.provider}' encountered in translate method."
        )
        return None
