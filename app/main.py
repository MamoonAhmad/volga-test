"""Transcription API.

POST /v1/transcriptions        upload audio -> 202 + job id
GET  /v1/transcriptions/{id}   job status and timestamped segments
GET  /health
"""
import hashlib
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.audio import SUPPORTED_FORMATS, InvalidAudio, probe_duration
from app.config import settings
from app.jobs import JobManager, JobStatus
from app.transcriber import SpeechEngine, build_engine

logging.basicConfig(level=logging.INFO)

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


def create_app(engine: SpeechEngine | None = None, *, max_upload_bytes: int | None = None,
               upload_dir: Path | None = None, backoff_base: float = 2.0) -> FastAPI:
    jobs = JobManager(engine or build_engine(settings), settings.max_workers,
                      settings.max_retries, backoff_base)
    max_bytes = max_upload_bytes or settings.max_upload_bytes
    uploads = upload_dir or settings.upload_dir

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="Transcription Service", version="1.0.0", lifespan=lifespan)

    @app.post("/v1/transcriptions", status_code=202, response_model=JobOut)
    async def create_transcription(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise HTTPException(415, f"Unsupported format {ext or '(none)'}; "
                                     f"supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
        path, sha256 = await save_upload(file, uploads, ext, max_bytes)
        try:
            probe_duration(path)  # reject corrupt files now, not later in the worker
        except InvalidAudio as e:
            path.unlink(missing_ok=True)
            raise HTTPException(422, str(e))

        job, created = jobs.submit(path, file.filename, sha256)
        if not created:
            path.unlink(missing_ok=True)  # duplicate upload: reuse the existing job
        return job

    @app.get("/v1/transcriptions/{job_id}", response_model=JobOut)
    def get_transcription(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": settings.engine}

    return app


app = create_app()
