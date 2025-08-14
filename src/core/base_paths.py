from pathlib import Path


class BasePaths:
    """Base class containing all path-related properties"""

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

    @property
    def LOGS_DIR(self) -> Path:
        """Directory path for application log files."""
        return self.BASE_DIR / "logs"

    @property
    def LOG_FILE(self) -> Path:
        """Full path to the main application log file."""
        return self.LOGS_DIR / "eora_ai_brag_assistant.log"

    @property
    def DATA_DIR(self) -> Path:
        """Base directory for all data files."""
        return self.PROJECT_ROOT / "data"

    @property
    def RAW_DATA_DIR(self) -> Path:
        """Directory for raw input data."""
        return self.DATA_DIR / "raw"

    @property
    def INPUT_CSV_PATH(self) -> Path:
        """Path to the main input CSV file."""
        return self.RAW_DATA_DIR / "eora_cases.csv"

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        """Directory for processed data files."""
        return self.DATA_DIR / "processed"

    @property
    def KNOWLEDGE_BASE_DIR(self) -> Path:
        """Directory for knowledge base files."""
        return self.DATA_DIR / "knowledge_base"

    @property
    def ENRICHED_DATA_PATH(self) -> Path:
        """Path to the enriched knowledge base chunks."""
        return self.PROCESSED_DATA_DIR / "knowledge_base_chunks.json"

    @property
    def SUMMARY_PATH(self) -> Path:
        """Path to the general summary file."""
        return self.KNOWLEDGE_BASE_DIR / "general_summary.txt"

    @property
    def VECTOR_STORE_PATH(self) -> Path:
        """Path to the vector store directory."""
        return self.KNOWLEDGE_BASE_DIR / "vector_store"
