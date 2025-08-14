import asyncio
import json
import random
from typing import List, cast, Dict, Any

from bs4 import BeautifulSoup, Tag
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

from src.core.config import settings
from src.core.logging_setup import log
from src.core.models import CaseStudyRecord
from src.core.models import KnowledgeBaseChunk


async def fetch_one(
    client: httpx.AsyncClient, case: CaseStudyRecord
) -> tuple[CaseStudyRecord, str | None]:
    """Asynchronously fetches HTML for a single case study, now with a delay."""
    url = case.url

    # --- Wait for a random, short duration before each request ---
    await asyncio.sleep(random.uniform(1, 3))  # Wait for 1 to 3 seconds

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        log.warning(f"Skipping invalid or malformed URL found in CSV: {url}")
        return case, None

    try:
        response = await client.get(url, timeout=20)

        if response.status_code == 200:
            log.info(f"Successfully fetched {url} (status: {response.status_code})")
            return case, response.text
        else:
            log.warning(
                f"Failed to fetch {url}, final status code: {response.status_code}"
            )
            return case, None

    except httpx.RequestError as e:
        log.error(f"HTTP request error for {url}: {e}")
        return case, None
    except Exception as e:
        log.error(f"An unexpected error occurred for {url}: {e}")
        return case, None


def extract_and_clean_text(html: str) -> str:
    """
    Surgically extracts meaningful text from EORA case study pages by targeting
    specific content blocks and excluding all known noise.
    """
    soup = BeautifulSoup(html, "html.parser")
    content_blocks = soup.find_all("div", class_="r t-rec")
    valuable_texts = []

    # --- Consolidated and Expanded Stop Phrase List ---
    # Include all phrases that indicate a non-content block.
    STOP_PHRASES = [
        "поможем подобрать решение",
        "нажимая на кнопку",
        "политикой в отношении обработки",
        "обсудить проект",
        "оставить заявку",
        "о компании",
        "портфолио",
        "разработчик проекта",
        "интересный факт",
        "услуги",
        "эмоции",
        "а что дальше",
        "команда проекта",
        "похожие проекты",
        "используем cookies",
    ]

    for block in content_blocks:
        # RULE 1: The most definitive check. If it's a form, discard immediately.
        if isinstance(block, Tag) and block.find("form"):
            continue

        # Get the clean text from the block just once for efficiency.
        current_block_text = "\n".join(block.stripped_strings)

        # RULE 2: Check for minimum length early. If it's too short, discard.
        if len(current_block_text) < 50:
            continue

        # RULE 3: Perform all text-based checks case-insensitively.
        current_block_text_lower = current_block_text.lower()
        if any(phrase in current_block_text_lower for phrase in STOP_PHRASES):
            continue

        # If the block survives all the checks, it's valuable. Add it.
        valuable_texts.append(current_block_text)

    return "\n\n".join(valuable_texts)


async def main() -> None:
    """Main async function to run the data enrichment pipeline."""
    log.info("Starting data enrichment process...")
    settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(
            settings.INPUT_CSV_PATH,
            header=None,
            names=["id", "title", "category", "description", "url"],
        )
        case_studies = [
            CaseStudyRecord(
            **cast(Dict[str, Any], row)
        ) for row in df.to_dict(orient="records")
        ]
        log.info(f"Successfully loaded {len(case_studies)} records from CSV.")

    except FileNotFoundError:
        log.error(
            f"Input file not found at {settings.INPUT_CSV_PATH}."
            f" Please place your CSV there."
        )
        return

    all_chunks: List[KnowledgeBaseChunk] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [fetch_one(client, case) for case in case_studies]
        results = await asyncio.gather(*tasks)

    log.info("All URLs have been processed. Now chunking text...")

    for case, html in results:
        if not html:
            # The fetch_one function logs the reason, so we just continue.
            continue

        full_text = extract_and_clean_text(html)
        if not full_text:
            log.warning(f"No text extracted from {case.url}")
            continue

        chunks = text_splitter.split_text(full_text)
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{case.id}_chunk_{i}"
            kb_chunk = KnowledgeBaseChunk(
                chunk_id=chunk_id,
                source_url=case.url,
                source_title=case.title,
                text=chunk_text,
                metadata={"category": case.category},
            )
            all_chunks.append(kb_chunk)

    log.info(f"Total chunks created: {len(all_chunks)}")

    chunks_as_dicts = [chunk.model_dump() for chunk in all_chunks]
    with open(settings.ENRICHED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_as_dicts, f, ensure_ascii=False, indent=4)

    log.info(f"Enriched data successfully saved to {settings.ENRICHED_DATA_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
