from enum import Enum

from src.ai_engine.generator import LLMGenerator
from src.core.logging_setup import log


class QueryType(Enum):
    GENERAL = "general"
    SPECIFIC = "specific"


class QueryRouter:
    """
    Classifies a user's query as either 'general' or 'specific' using an LLM.
    """

    def __init__(self, generator: LLMGenerator):
        """
        Initializes the router with a pre-configured LLM generator.

        Args:
            generator: An instance of the LLMGenerator class.
        """
        self.generator = generator
        self.system_prompt = """
        You are an expert query classification system. Your only task is \
        to classify the user's question into one of two categories:
        1. 'general': For questions about the company EORA itself, its activities, its \
        purpose, or what it does overall. Examples: "What is EORA?", "What are your \
        services?", "Tell me about your company."
        2. 'specific': For questions asking how EORA can help a specific industry, \
        solve a particular problem, or for examples of their work. Examples: \
        "What can you do for retail?", "How can you help my taxi company?", \
        "Do you have experience with computer vision?"

        You must respond with ONLY the word 'general' or 'specific' and nothing else.
        """

    def route(self, query: str) -> QueryType:
        """
        Routes the user's query to the appropriate category.

        Args:
            query: The user's question.

        Returns:
            Either 'general' or 'specific'.
        """
        log.info(f"Routing query: '{query}'")

        # We create a simple prompt by combining the system message and the user query
        full_prompt = f'{self.system_prompt}\n\nUser Question: "{query}"'

        try:
            raw_response = self.generator.generate(full_prompt)

            if raw_response and isinstance(raw_response, str):
                response = raw_response.strip().lower()
                if "general" in response:
                    return QueryType.GENERAL
                else:
                    return (
                        QueryType.SPECIFIC
                    )
            else:
                log.warning(
                    "Query router received no response from LLM. Defaulting to 'specific'."
                )
                return QueryType.SPECIFIC
        except Exception as e:
            log.error(f"Failed to route query due to an error: {e}")
            return QueryType.SPECIFIC


# --- Simple Test Block ---z
if __name__ == "__main__":
    try:
        # We need to initialize the components it depends on
        llm_generator = LLMGenerator()
        router = QueryRouter(generator=llm_generator)

        print("\n--- Testing Router ---")

        query1 = "Что вы можете сделать для ритейла?"
        route1 = router.route(query1)
        print(f"Query: '{query1}' -> Route: {route1}")  # Expected: specific

        query2 = "Чем занимается ваша компания?"
        route2 = router.route(query2)
        print(f"Query: '{query2}' -> Route: {route2}")  # Expected: general

        query3 = "Расскажите об EORA"
        route3 = router.route(query3)
        print(f"Query: '{query3}' -> Route: {route3}")  # Expected: general

        query4 = "Есть ли у вас опыт в логистике?"
        route4 = router.route(query4)
        print(f"Query: '{query4}' -> Route: {route4}")  # Expected: specific

    except Exception as e:
        log.error(f"Failed to run router test: {e}")
