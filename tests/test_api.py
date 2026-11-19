import dataclasses
import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app
from app.transcriber import MockEngine, Segment, TranscriptionError, TransientError
from tests.conftest import make_wav


class FlakyEngine:
    """Fails transiently `failures` times, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def transcribe(self, path, duration):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientError("503 from provider")
        return [Segment(0.0, 1.0, "recovered", 0.9)]


class BrokenEngine:
    def __init__(self):
        self.calls = 0

    def transcribe(self, path, duration):
        self.calls += 1
        raise TranscriptionError("invalid credentials")


@pytest.fixture
def make_client(tmp_path):
    def _make(engine=None, **overrides):
        defaults = {"upload_dir": tmp_path, "api_key": None, "max_retries": 3}
        cfg = dataclasses.replace(settings, **{**defaults, **overrides})
        return TestClient(create_app(engine or MockEngine(), settings=cfg, backoff_base=0.01))
    return _make


def upload(client, data: bytes, name: str = "sample.wav", headers: dict | None = None):
    return client.post("/v1/transcriptions", files={"file": (name, data, "audio/wav")},
                       headers=headers)


def wait_for(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/transcriptions/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def test_upload_returns_job_and_timestamped_segments(make_client, wav_bytes):
    client = make_client()
    resp = upload(client, wav_bytes)
    assert resp.status_code == 202
    job = wait_for(client, resp.json()["id"])
    assert job["status"] == "done"
    assert job["duration_sec"] == 1.0
    assert job["segments"][0] == {"start": 0.0, "end": 2.5,
                                  "text": "Hello, this is a test recording.", "confidence": 0.98}


def test_rejects_unsupported_format(make_client):
    resp = upload(make_client(), b"hello", name="notes.txt")
    assert resp.status_code == 415


def test_rejects_empty_file(make_client):
    assert upload(make_client(), b"").status_code == 400


def test_rejects_corrupt_audio(make_client):
    resp = upload(make_client(), b"this is not audio" * 100)
    assert resp.status_code == 422


def test_rejects_oversized_file(make_client, wav_bytes):
    resp = upload(make_client(max_upload_bytes=100), wav_bytes)
    assert resp.status_code == 413


def test_unknown_job_is_404(make_client):
    assert make_client().get("/v1/transcriptions/nope").status_code == 404


def test_duplicate_upload_reuses_job(make_client, wav_bytes):
    client = make_client()
    first = upload(client, wav_bytes).json()
    second = upload(client, wav_bytes).json()
    assert first["id"] == second["id"]


def test_transient_errors_are_retried(make_client, wav_bytes):
    engine = FlakyEngine(failures=2)
    client = make_client(engine)
    job = wait_for(client, upload(client, wav_bytes).json()["id"])
    assert job["status"] == "done"
    assert job["attempts"] == 3
    assert job["segments"][0]["text"] == "recovered"


def test_gives_up_after_max_retries(make_client, wav_bytes):
    client = make_client(FlakyEngine(failures=99))
    job = wait_for(client, upload(client, wav_bytes).json()["id"])
    assert job["status"] == "failed"
    assert "Gave up after 3 attempts" in job["error"]


def test_permanent_errors_are_not_retried(make_client, wav_bytes):
    engine = BrokenEngine()
    client = make_client(engine)
    job = wait_for(client, upload(client, wav_bytes).json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == "invalid credentials"
    assert engine.calls == 1


def test_failed_job_can_be_resubmitted(make_client, wav_bytes):
    client = make_client(FlakyEngine(failures=3))  # first job exhausts 3 attempts, retry succeeds
    first = wait_for(client, upload(client, wav_bytes).json()["id"])
    assert first["status"] == "failed"
    second = wait_for(client, upload(client, wav_bytes).json()["id"])
    assert second["id"] != first["id"]
    assert second["status"] == "done"


def test_concurrent_uploads(make_client):
    client = make_client()
    ids = [upload(client, make_wav(seconds=0.1 * (i + 1))).json()["id"] for i in range(10)]
    assert len(set(ids)) == 10
    assert all(wait_for(client, i)["status"] == "done" for i in ids)


def test_audio_is_deleted_after_processing(make_client, wav_bytes, tmp_path):
    client = make_client()
    wait_for(client, upload(client, wav_bytes).json()["id"])
    assert list(tmp_path.iterdir()) == []


def test_audio_is_deleted_after_failure(make_client, wav_bytes, tmp_path):
    client = make_client(BrokenEngine())
    wait_for(client, upload(client, wav_bytes).json()["id"])
    assert list(tmp_path.iterdir()) == []


def test_rejected_upload_leaves_no_file(make_client, tmp_path):
    upload(make_client(), b"not audio" * 100)
    assert list(tmp_path.iterdir()) == []


def test_unexpected_engine_crash_marks_job_failed(make_client, wav_bytes):
    class CrashingEngine:
        def transcribe(self, path, duration):
            raise RuntimeError("bug")

    client = make_client(CrashingEngine())
    job = wait_for(client, upload(client, wav_bytes).json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == "Internal error during transcription"  # internals not leaked


def test_api_key_required_when_configured(make_client, wav_bytes):
    client = make_client(api_key="secret")
    assert upload(client, wav_bytes).status_code == 401
    assert upload(client, wav_bytes, headers={"X-API-Key": "wrong"}).status_code == 401
    resp = upload(client, wav_bytes, headers={"X-API-Key": "secret"})
    assert resp.status_code == 202
    job_id = resp.json()["id"]
    assert client.get(f"/v1/transcriptions/{job_id}").status_code == 401
    assert client.get(f"/v1/transcriptions/{job_id}",
                      headers={"X-API-Key": "secret"}).status_code == 200


def test_health_is_public(make_client):
    resp = make_client(api_key="secret").get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
