"""Transcription API.

POST /v1/transcriptions        upload audio -> 202 + job id
GET  /v1/transcriptions/{id}   job status and timestamped segments
GET  /health
"""
import hashlib
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.audio import SUPPORTED_FORMATS, InvalidAudio, probe_duration
from app.config import Settings
from app.config import settings as default_settings
from app.jobs import JobManager, JobStatus
from app.transcriber import SpeechEngine, build_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CHUNK_SIZE = 1024 * 1024


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    confidence: float | None = None


class JobOut(BaseModel):
    id: str
    filename: str
    status: JobStatus
    attempts: int
    duration_sec: float | None = None
    segments: list[SegmentOut] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


async def save_upload(file: UploadFile, dest_dir: Path, ext: str, max_bytes: int) -> tuple[Path, str]:
    """Stream the upload to disk in chunks (never fully in memory), hashing as we go."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"
    digest, size = hashlib.sha256(), 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File exceeds {max_bytes // (1024 * 1024)} MB limit")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest, digest.hexdigest()


def create_app(engine: SpeechEngine | None = None, *, settings: Settings | None = None,
               backoff_base: float = 2.0) -> FastAPI:
    cfg = settings or default_settings
    jobs = JobManager(engine or build_engine(cfg), cfg.max_workers, cfg.max_retries, backoff_base)

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: str | None = Security(api_key_header)) -> None:
        if cfg.api_key and not (key and secrets.compare_digest(key, cfg.api_key)):
            raise HTTPException(401, "Missing or invalid API key")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="Transcription Service", version="1.0.0", lifespan=lifespan)

    @app.post("/v1/transcriptions", status_code=202, response_model=JobOut,
              dependencies=[Depends(require_api_key)])
    async def create_transcription(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise HTTPException(415, f"Unsupported format {ext or '(none)'}; "
                                     f"supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
        path, sha256 = await save_upload(file, cfg.upload_dir, ext, cfg.max_upload_bytes)
        try:
            probe_duration(path)  # reject corrupt files now, not later in the worker
        except InvalidAudio as e:
            path.unlink(missing_ok=True)
            raise HTTPException(422, str(e)) from e

        job, created = jobs.submit(path, file.filename, sha256)
        if not created:
            path.unlink(missing_ok=True)  # duplicate upload: reuse the existing job
        return job

    @app.get("/v1/transcriptions/{job_id}", response_model=JobOut,
             dependencies=[Depends(require_api_key)])
    def get_transcription(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": cfg.engine}

    return app


app = create_app()
