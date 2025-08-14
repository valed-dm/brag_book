import importlib.util
from typing import Dict
from typing import List
from typing import Optional

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import Pipeline
from transformers import pipeline

from src.core.logging_setup import log


class LLMTranslator:
    """
    Handles translation of a text from a source language to a target language
    using specified Hugging Face language models.
    """

    DEFAULT_MODEL_MAPPING: Dict[str, str] = {
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    }

    def __init__(
        self,
        model_mapping: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ):
        """Initializes the LLMTranslator."""
        self.model_mapping = (
            model_mapping if model_mapping else self.DEFAULT_MODEL_MAPPING
        )
        self.device_to_use = device
        self.translation_pipelines: Dict[str, Optional[Pipeline]] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Loads the translation models and tokenizers."""
        _safetensors_available = importlib.util.find_spec("safetensors") is not None
        if not _safetensors_available:
            log.warning("LLMTranslator: 'safetensors' library not found.")

        for lang_pair, model_name in self.model_mapping.items():
            try:
                log.info(
                    f"LLMTranslator: Loading model '{model_name}' for '{lang_pair}'."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
                model_obj = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, use_safetensors=_safetensors_available
                )

                loaded_pipeline = pipeline(
                    "translation",
                    model=model_obj,
                    tokenizer=tokenizer,
                    device=self.device_to_use,
                )
                self.translation_pipelines[lang_pair] = loaded_pipeline
                log.info(
                    f"LLMTranslator: Model '{model_name}' loaded successfully"
                    f" on device '{loaded_pipeline.device}'."
                )
            except Exception as e:
                log.exception(
                    f"LLMTranslator: Error loading model '{model_name}': {e}."
                )
                self.translation_pipelines[lang_pair] = None

    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translates the given query text.

        Returns the translated string, or None if translation is not possible.
        """
        if not text or source_lang == target_lang:
            # Return None for empty input to be consistent with "no translation"
            return None if not text else text

        lang_pair_key = f"{source_lang}-{target_lang}"
        selected_pipeline = self.translation_pipelines.get(lang_pair_key)

        if selected_pipeline is None:
            log.error(
                f"LLMTranslator: No model available for '{lang_pair_key}'."
                f" Cannot translate."
            )
            return None

        try:
            # The pipeline returns a list of dicts, e.g., [{'translation_text': '...'}]
            translated_output: List[Dict[str, str]] = selected_pipeline(text)

            # Check for a valid, non-empty list response
            if translated_output and isinstance(translated_output, list):
                translation = translated_output[0].get("translation_text")
                return translation.strip() if translation else None

            log.warning("LLMTranslator: Translation output format was unexpected.")
            return None

        except Exception as e:
            log.exception(f"LLMTranslator: An error occurred during translation: {e}")
            return None
