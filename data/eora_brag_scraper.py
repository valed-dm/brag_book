import csv
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from bs4 import Tag
import httpx


try:
    from src.core.logging_setup import log
except ImportError:
    from loguru import logger as log


class EoraScraper:
    """
    A class to fetch, enrich, and save case study data from EORA's website and API.

    This scraper performs a two-step process:
    1. Fetches a master list of cases from the Tilda API.
    2. Enriches this data by scraping the description from each individual case page.
    """

    EORA_BASE_URL = "https://eora.ru"
    API_BASE_URL = "https://store.tildaapi.com/api/getproductslist/"
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    # Default list of API blocks to scrape
    DEFAULT_API_BLOCKS = [
        ("511384921851", "573614930", 9),  # Main case list
        ("911458483521", "570167622", 9),  # Another category
    ]

    def __init__(
        self,
        output_path: Path,
        api_blocks: Optional[List[Tuple[str, str, int]]] = None,
    ):
        """
        Initializes the scraper with a specific output path and API blocks.

        Args:
            output_path (Path): The file path to save the final CSV.
            api_blocks (list, optional): A list of API block tuples.
                                         Defaults to DEFAULT_API_BLOCKS.
        """
        self.output_path = output_path
        api_blocks_provided = api_blocks is not None
        self.api_blocks = api_blocks if api_blocks_provided else self.DEFAULT_API_BLOCKS
        self.cases: List[Dict[str, Any]] = []

    def _fetch_base_case_data_from_api(self) -> None:
        """
        Step 1: Fetches the primary list of all cases from the Tilda API.
        Populates self.cases with the initial data.
        """
        if not self.api_blocks:
            log.warning("No API blocks to scrape. Exiting.")
            return  # Exit the method

        log.info("--- Step 1: Fetching master list of cases from API ---")
        seen_uids = set()

        with httpx.Client(timeout=20) as client:
            for storepartuid, recid, size in self.api_blocks:
                page = 1
                while True:
                    params: Dict[str, Union[str, int]] = {
                        "storepartuid": storepartuid,
                        "recid": recid,
                        "slice": page,
                        "getparts": "true",
                        "size": size,
                    }
                    try:
                        r = client.get(
                            self.API_BASE_URL,
                            params=params,
                            headers=self.HEADERS,
                        )
                        r.raise_for_status()
                        data = r.json()
                    except Exception as e:
                        log.error(f"Failed to fetch API data for block {recid}: {e}")
                        break

                    products = data.get("products", [])
                    if not products:
                        break

                    for case in products:
                        if (uid := case.get("uid")) and uid not in seen_uids:
                            link = case.get("buttonlink") or case.get("url", "")
                            full_url = (
                                urljoin(
                                    self.EORA_BASE_URL,
                                    link,
                                )
                                if link and link.startswith("/")
                                else link
                            )

                            self.cases.append(
                                {
                                    "uid": uid,
                                    "title": case.get("title"),
                                    "category": case.get("mark"),
                                    "full_url": full_url,
                                    "description": "",  # To be filled in Step 2
                                }
                            )
                            seen_uids.add(uid)
                    page += 1

        log.info(f"API fetch complete. Found {len(self.cases)} unique cases.")

    def _enrich_cases_with_descriptions(self) -> None:
        """
        Step 2: Scrapes the individual page of each case in self.cases
        to get its meta description and updates the data in place.
        """
        log.info("--- Step 2: Enriching cases with descriptions from their pages ---")
        if not self.cases:
            log.warning("No cases to enrich. Skipping description scraping.")
            return

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            for i, case in enumerate(self.cases, 1):
                url = case.get("full_url")
                if not url:
                    continue

                log.info(
                    f"[{i}/{len(self.cases)}] "
                    f"Scraping description for: {case['title']}"
                )
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")

                    desc_tag = soup.find("meta", attrs={"name": "description"})
                    if isinstance(desc_tag, Tag) and "content" in desc_tag.attrs:
                        content_value = desc_tag["content"]
                        final_description = ""
                        if isinstance(content_value, list):
                            # If it's a list, join all parts into a single string
                            final_description = " ".join(content_value)
                        elif isinstance(content_value, str):
                            # If it's already a string, use it directly
                            final_description = content_value
                        case["description"] = final_description.strip()
                    else:
                        log.warning(f"  -> Meta description not found for {url}")

                except httpx.RequestError as e:
                    log.error(f"  -> Request failed for {url}: {e}")
                except Exception as e:
                    log.error(f"  -> An unexpected error occurred for {url}: {e}")

    def _save_to_csv(self) -> None:
        """
        Saves the final, enriched data from self.cases to the CSV file.
        """
        log.info("--- Step 3: Saving all data to CSV ---")
        if not self.cases:
            log.warning("No cases to save.")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["uid", "title", "category", "description", "full_url"]

        with self.output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.cases)

        log.info(f"Successfully saved {len(self.cases)} cases to {self.output_path}")

    def run(self) -> None:
        """
        Executes the full scrape and save process.
        """
        self._fetch_base_case_data_from_api()
        self._enrich_cases_with_descriptions()
        self._save_to_csv()


if __name__ == "__main__":
    output_file_path = Path("raw") / "eora_cases.csv"
    scraper = EoraScraper(output_path=output_file_path)
    scraper.run()
