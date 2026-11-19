"""Job tracking and background processing with bounded concurrency and retries.

In-memory for this exercise. In production the store would be Postgres/Redis and the
thread pool a queue (Celery/SQS) so jobs survive restarts and workers scale separately.
"""
import logging
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from app.audio import InvalidAudio, normalize, probe_duration
from app.transcriber import Segment, SpeechEngine, TranscriptionError, TransientError

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Job:
    id: str
    filename: str
    sha256: str
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    duration_sec: float | None = None
    segments: list[Segment] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


class JobManager:
    def __init__(self, engine: SpeechEngine, max_workers: int, max_retries: int,
                 backoff_base: float = 2.0):
        self.engine = engine
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._jobs: dict[str, Job] = {}
        self._by_hash: dict[str, str] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stt")

    def submit(self, path: Path, filename: str, sha256: str) -> tuple[dict, bool]:
        """Queue a job. Identical files reuse the existing job unless it failed.

        Returns (job, created).
        """
        with self._lock:
            existing_id = self._by_hash.get(sha256)
            if existing_id and self._jobs[existing_id].status != JobStatus.FAILED:
                return asdict(self._jobs[existing_id]), False
            job = Job(id=uuid.uuid4().hex, filename=filename, sha256=sha256)
            self._jobs[job.id] = job
            self._by_hash[sha256] = job.id
            snapshot = asdict(job)
        self._pool.submit(self._run, job.id, path)
        return snapshot, True

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job else None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def _run(self, job_id: str, path: Path) -> None:
        audio = path
        try:
            self._update(job_id, status=JobStatus.PROCESSING)
            audio = normalize(path)
            self._transcribe(job_id, audio)
        except InvalidAudio as e:
            self._update(job_id, status=JobStatus.FAILED, error=str(e))
        except Exception:
            log.exception("job %s crashed", job_id)
            self._update(job_id, status=JobStatus.FAILED, error="Internal error during transcription")
        finally:
            # The transcript is the product; raw audio is not retained after processing.
            path.unlink(missing_ok=True)
            if audio != path:
                audio.unlink(missing_ok=True)

    def _transcribe(self, job_id: str, audio: Path) -> None:
        duration = probe_duration(audio)
        self._update(job_id, duration_sec=duration)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._update(job_id, attempts=attempt)
            try:
                segments = self.engine.transcribe(audio, duration)
            except TransientError as e:
                last_error = e
                log.warning("job %s attempt %d failed transiently: %s", job_id, attempt, e)
                if attempt < self.max_retries:
                    # Exponential backoff with jitter so concurrent retries don't align.
                    time.sleep(self.backoff_base ** attempt + random.uniform(0, self.backoff_base))  # noqa: S311
                continue
            except TranscriptionError as e:
                self._update(job_id, status=JobStatus.FAILED, error=str(e))
                return
            self._update(job_id, status=JobStatus.DONE, segments=segments)
            return

        self._update(job_id, status=JobStatus.FAILED,
                     error=f"Gave up after {self.max_retries} attempts: {last_error}")
