from typing import Any, Union
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


ChromaMetadataValue = Union[str, int, float, bool, None]

# Represents a single case study from the initial
# data/raw/eora_cases.csv
class CaseStudyRecord(BaseModel):
    id: str = Field(..., description="Unique identifier for the case study.")
    title: str = Field(..., description="The title of the case study.")
    category: str = Field(
        ..., description="The primary activity category (e.g., 'Зрение', 'Чат-бот')."
    )
    description: str = Field(..., description="A short summary of the project.")
    url: str = Field(..., description="The URL to the full case study page.")


# Represents a single chunk of text prepared for the knowledge base
# This is the main data structure for the RAG system
class KnowledgeBaseChunk(BaseModel):
    chunk_id: str = Field(
        ..., description="Unique ID for the chunk (e.g., 'case_id_chunk_0')."
    )
    source_url: str = Field(..., description="The URL of the source case study.")
    source_title: str = Field(..., description="The title of the source case study.")
    text: str = Field(..., description="The actual text content of the chunk.")
    # We can add more metadata later if needed, like the original category
    metadata: Optional[Dict[str, ChromaMetadataValue]] = Field(
        default_factory=dict, description="Any other relevant metadata."
    )
