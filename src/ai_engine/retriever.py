from typing import Any
from typing import Dict
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.logging_setup import log


class Retriever:
    """
    Handles the retrieval of relevant document chunks from the ChromaDB vector store.
    """

    def __init__(self) -> None:
        """
        Initializes the Retriever by loading the embedding model and connecting
        to the vector store.
        """
        try:
            # Load the embedding model that was used to create the embeddings
            log.info(
                f"Loading embedding model for retriever:"
                f" {settings.EMBEDDING_MODEL_NAME}"
            )
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

            # Connect to the persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(settings.VECTOR_STORE_PATH)
            )

            # Get the collection
            self.collection_name = "eora_cases"
            self.collection = self.client.get_collection(name=self.collection_name)
            self.distance_threshold: float = settings.CHROMA_DISTANCE_THRESHOLD
            log.info(
                f"Retriever initialized with distance threshold:"
                f" {self.distance_threshold}"
            )

        except Exception as e:
            log.error(f"Failed to initialize Retriever: {e}")
            # This is a critical error, so we re-raise it to stop the application
            # from starting in a broken state.
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant document chunks for a given query.

        Args:
            query: The user's question or search term.
            top_k: The number of top results to retrieve.

        Returns:
            A list of dictionaries, where each dictionary contains the retrieved
            document text and its metadata.
        """
        if not query:
            return []

        log.info(f"Performing search for query: '{query}'")

        # 1. Convert the user's query into a vector embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # 2. Query the collection in ChromaDB
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )

            # 1. Safely get the top-level lists from the results dictionary.
            #    Use .get() to avoid KeyErrors and default to an empty list.
            metadatas_list = results.get("metadatas") or []
            documents_list = results.get("documents") or []
            distances_list = results.get("distances") or []

            # 2. Check if we actually got any results back for our query.
            #    Since we sent one query, we expect one list of results back.
            if not metadatas_list or not documents_list or not distances_list:
                log.warning("ChromaDB query returned no results.")
                return []

            # We are only interested in the results for our single query (at index 0)
            top_metadatas = metadatas_list[0]
            top_documents = documents_list[0]
            top_distances = distances_list[0]

            # 3. Now, iterate safely to build the final list.
            retrieved_chunks = []
            # Use zip to elegantly iterate over all three lists at once.
            # The length will be determined by the shortest list, which is inherently safe.
            for i, (metadata, doc, dist) in enumerate(
                zip(top_metadatas, top_documents, top_distances)
            ):
                # We can add a distance filter here if we want
                if dist > self.distance_threshold:
                    continue

                retrieved_chunks.append(
                    {
                        "id": i,
                        "text": doc,
                        "distance": dist,
                        # 'metadata' is already a single dict here, so no indexing is needed.
                        "metadata": metadata,
                    }
                )

            return retrieved_chunks

        except Exception as e:
            log.exception(f"An error occurred during retrieval from ChromaDB: {e}")
            return []


# --- Simple Test Block ---
if __name__ == "__main__":
    try:
        retriever = Retriever()

        # Test query
        test_query = "Что вы можете сделать для такси?"

        log.info(f"\n--- Testing Retriever with query: '{test_query}' ---")
        search_results = retriever.search(test_query)

        if search_results:
            print(f"\nFound {len(search_results)} results:")
            for i, result in enumerate(search_results):
                print(f"\n--- Result {i + 1} ---")
                print(f"Source: {result['metadata'].get('source_title', 'N/A')}")
                print(f"URL: {result['metadata'].get('source_url', 'N/A')}")
                # Print the first 150 characters of the text
                print(f"Text Snippet: {result['text'][:150]}...")
        else:
            print("No results found.")

    except Exception as e:
        log.error(f"Failed to run retriever test: {e}")
