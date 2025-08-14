import json
from typing import Mapping, List, cast

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.logging_setup import log
from src.core.models import KnowledgeBaseChunk, ChromaMetadataValue


def main() -> None:
    """Main function to build and save the vector store."""
    log.info("Starting vector store build process...")

    # 1. Load the processed data chunks
    try:
        with open(settings.ENRICHED_DATA_PATH, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            chunks = [KnowledgeBaseChunk(**data) for data in chunks_data]
        log.info(
            f"Successfully loaded {len(chunks)} chunks from"
            f" {settings.ENRICHED_DATA_PATH}"
        )
    except FileNotFoundError:
        log.error(
            f"Processed chunks file not found at {settings.ENRICHED_DATA_PATH}."
            f" Please run 01_scrape_and_enrich.py first."
        )
        return
    except Exception as e:
        log.error(f"Error loading or parsing chunks file: {e}")
        return

    # 2. Initialize the embedding model
    # This model runs locally and will be downloaded on first use.
    try:
        log.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME, device="cpu"
        )  # Specify CPU for broad compatibility
        log.info("Embedding model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load sentence-transformer model: {e}")
        return

    # 3. Initialize the ChromaDB client
    # This creates a persistent database at the specified path.
    client = chromadb.PersistentClient(path=str(settings.VECTOR_STORE_PATH))

    # Create a new collection. If it already exists, you might want to delete it first
    # for a clean build.
    collection_name = "eora_cases"
    if collection_name in [c.name for c in client.list_collections()]:
        log.warning(
            f"Collection '{collection_name}' already exists."
            f" Deleting it for a fresh build."
        )
        client.delete_collection(name=collection_name)

    collection = client.create_collection(name=collection_name)
    log.info(f"Created ChromaDB collection: '{collection_name}'")

    # 4. Prepare data for ChromaDB
    # ChromaDB requires lists of IDs, documents (the text), and metadatas.
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source_url": chunk.source_url,
            "source_title": chunk.source_title,
            **(chunk.metadata or {}),
        }
        for chunk in chunks
    ]

    # 5. Generate embeddings for all documents
    # This is the most time-consuming step.
    log.info(
        f"Generating embeddings for {len(documents)} documents."
        f" This may take a while..."
    )
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    log.info("Embeddings generated successfully.")

    # 6. Add the data to the ChromaDB collection
    # Chroma handles batching automatically.
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=cast(List[Mapping[str, ChromaMetadataValue]], metadatas),
        )
        log.info(f"Successfully added {collection.count()} items to the vector store.")
    except Exception as e:
        log.error(f"An error occurred while adding data to ChromaDB: {e}")
        return

    log.info(
        f"Vector store build process complete."
        f" Database saved at: {settings.VECTOR_STORE_PATH}"
    )


if __name__ == "__main__":
    main()
