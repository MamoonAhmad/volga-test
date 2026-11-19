import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # real environment variables take precedence over .env


@dataclass(frozen=True)
class Settings:
    engine: str = os.getenv("STT_ENGINE", "google")  # google | mock
    language: str = os.getenv("STT_LANGUAGE", "en-US")
    gcp_project: str | None = os.getenv("GOOGLE_CLOUD_PROJECT")
    gcs_bucket: str | None = os.getenv("GCS_BUCKET")  # required for audio > ~1 min on Google
    api_key: str | None = os.getenv("API_KEY") or None  # unset = auth disabled
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))


settings = Settings()
