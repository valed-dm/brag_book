from typing import Optional, List, Dict, Any

from src.ai_engine.generator import LLMGenerator
from src.ai_engine.query_router import QueryRouter, QueryType
from src.ai_engine.retriever import Retriever
from src.ai_engine.translator import LLMTranslator
from src.core.config import settings
from src.core.logging_setup import log


class Orchestrator:
    """Coordinates the entire query-answering process."""

    def __init__(self) -> None:
        """Initializes all necessary AI components with graceful fallbacks."""
        log.info("Initializing Orchestrator and its components...")

        try:
            self.generator = LLMGenerator()
            self.retriever = Retriever()
            self.router = QueryRouter(self.generator)
            with open(settings.SUMMARY_PATH, "r", encoding="utf-8") as f:
                self.general_summary = f.read()
        except Exception as e:
            log.error(f"Failed to initialize CRITICAL components: {e}")
            raise

        self.translator = None
        if settings.TRANSLATION_PROVIDER == "specialized":
            try:
                self.translator = LLMTranslator(
                    model_mapping=settings.TRANSLATOR_MODEL_MAPPING
                )
                log.info("Initialized successfully with SPECIALIZED translator.")
            except Exception as e:
                log.warning(
                    f"Failed to initialize SPECIALIZED translator: {e}."
                    f" Will fall back to GENERATOR for translation."
                )
        else:
            log.info("Using GENERATOR for translation (as configured).")

        log.info("Orchestrator initialization complete.")

    def answer(self, query: str, output_language: str = "english") -> Optional[str]:
        """Answers a user's query and provides a high-quality response."""
        query_type = self.router.route(query)

        if query_type == QueryType.GENERAL:
            # For general queries, the summary is already complete.
            # We can translate it if needed.
            base_summary = self.general_summary
            if output_language.lower() != "english":
                return self._translate_text(base_summary, output_language)
            return base_summary

        context_docs = self.retriever.search(query)
        if not context_docs:
            return (
                "Unfortunately, I could not find relevant information for your request."
            )

        # 1. Generate the prose part of the answer
        prompt = self._build_rag_prompt(query, context_docs)
        prose_answer: Optional[str] = self.generator.generate(prompt)
        if not prose_answer:
            log.error("Failed to generate an answer for the query.")
            return "Sorry, I was unable to generate an answer for your question."

        # 2. Build the source list string
        source_list = self._build_source_list(context_docs)

        # 3. Translate ONLY the prose if needed
        if output_language.lower() != "english":
            translated_prose = self._translate_text(prose_answer, output_language)
            final_prose = translated_prose or prose_answer
        else:
            final_prose = prose_answer

        return f"{final_prose}\n\n{source_list}"

    def _translate_text(self, text: str, language: str) -> Optional[str]:
        """Helper function to call the correct translation engine."""
        if self.translator:
            log.info(f"Translating text to {language} with SPECIALIZED provider...")
            return self.translator.translate(
                text, source_lang="en", target_lang=language.lower()
            )
        else:
            log.info(f"Translating text to {language} with GENERATOR provider...")
            return self.generator.translate(text, target_language=language)

    def _build_rag_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Builds a clean prompt for the generator with numbered sources."""
        context_str = ""
        for i, doc in enumerate(context):
            context_str += f"--- Источник [{i + 1}] ---\n{doc['text']}\n\n"

        prompt = f"""
        Ты — полезный ИИ-ассистент компании EORA. Твоя задача — отвечать на вопросы \
        потенциальных клиентов, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже \
        контексте.

        Контекст из кейсов:
        {context_str}

        Запрос клиента: "{query}"

        Инструкции:
        1.  Внимательно изучи пронумерованные источники.
        2.  Сформулируй полезный ответ на запрос клиента, демонстрируя возможности \
        EORA на основе примеров из контекста.
        3.  Критически важно: Когда ты используешь информацию из источника, ставь его \
        номер в квадратных скобках в конце предложения. Например: "Мы можем создать \
        чат-бота для ритейла [1]." или "Наш опыт в компьютерном зрении [3][5] \
        позволяет решать такие задачи."
        4.  Не придумывай информацию. Если в контексте нет прямого ответа, объясни, \
        как технологии из приведенных примеров могут быть применены для решения задачи \
        клиента.
        5.  Будь краток и отвечай по существу. НЕ включай список источников в свой \
        ответ, только номера в скобках.
        """

        return prompt

    def _build_source_list(self, context: List[Dict[str, Any]]) -> str:
        """Creates a formatted string of the sources used."""
        source_list_str = "Источники:\n"
        for i, doc in enumerate(context):
            metadata = doc.get("metadata") or {}
            title = metadata.get("source_title", "N/A")
            url = metadata.get("source_url", "N/A")

            source_list_str += f"[{i + 1}] {title}\n    {url}\n"

        return source_list_str
