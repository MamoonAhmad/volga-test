from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.api_core import exceptions as gexc

from app.config import Settings
from app.transcriber import (
    GoogleSTTEngine,
    MockEngine,
    TranscriptionError,
    TransientError,
    build_engine,
)


def word(start: float):
    return SimpleNamespace(start_offset=timedelta(seconds=start))


def result(text: str, end: float, words: list | None = None, confidence: float = 0.9):
    alt = SimpleNamespace(transcript=text, confidence=confidence, words=words or [])
    return SimpleNamespace(alternatives=[alt], result_end_offset=timedelta(seconds=end))


@pytest.fixture
def engine():
    """GoogleSTTEngine without a real client; network methods are stubbed per test."""
    e = GoogleSTTEngine.__new__(GoogleSTTEngine)
    e.bucket = "bucket"
    return e


class TestToSegments:
    def test_maps_results_to_timestamped_segments(self):
        segments = GoogleSTTEngine._to_segments([
            result(" Hello there. ", end=1.5, words=[word(0.12)], confidence=0.9512),
            result("How are you?", end=3.25, words=[word(1.8)]),
        ])
        assert [(s.start, s.end, s.text, s.confidence) for s in segments] == [
            (0.12, 1.5, "Hello there.", 0.951),
            (1.8, 3.25, "How are you?", 0.9),
        ]

    def test_start_falls_back_to_previous_end_without_word_offsets(self):
        segments = GoogleSTTEngine._to_segments([
            result("First.", end=2.0, words=[word(0.5)]),
            result("Second.", end=4.0),
        ])
        assert segments[1].start == 2.0

    def test_skips_empty_results(self):
        empty_alts = SimpleNamespace(alternatives=[], result_end_offset=timedelta(seconds=1))
        segments = GoogleSTTEngine._to_segments([
            empty_alts,
            result("   ", end=2.0),
            result("Kept.", end=3.0, words=[word(2.1)]),
        ])
        assert [s.text for s in segments] == ["Kept."]

    def test_no_results_returns_empty_list(self):
        assert GoogleSTTEngine._to_segments([]) == []


class TestRouting:
    @pytest.mark.parametrize("duration, expected", [
        (10.0, "sync"),
        (55.0, "sync"),
        (55.1, "batch"),
        (3600.0, "batch"),
        (None, "sync"),  # unknown duration: try sync
    ])
    def test_chooses_sync_or_batch_by_duration(self, engine, duration, expected):
        calls = []
        engine._sync = lambda path: calls.append("sync") or []
        engine._batch = lambda path: calls.append("batch") or []
        engine.transcribe(Path("a.wav"), duration)
        assert calls == [expected]

    def test_long_audio_without_bucket_is_permanent_error(self, engine):
        engine.bucket = None
        with pytest.raises(TranscriptionError, match="GCS_BUCKET") as exc:
            engine._batch(Path("a.wav"))
        assert not isinstance(exc.value, TransientError)


class TestErrorClassification:
    @pytest.mark.parametrize("error", [
        gexc.ServiceUnavailable("down"),
        gexc.DeadlineExceeded("slow"),
        gexc.ResourceExhausted("quota"),
        gexc.InternalServerError("boom"),
    ])
    def test_retryable_google_errors_become_transient(self, engine, error):
        engine._sync = lambda path: (_ for _ in ()).throw(error)
        with pytest.raises(TransientError):
            engine.transcribe(Path("a.wav"), 5.0)

    @pytest.mark.parametrize("error", [
        gexc.InvalidArgument("bad audio"),
        gexc.PermissionDenied("no access"),
        gexc.Unauthenticated("bad creds"),
    ])
    def test_client_google_errors_are_permanent(self, engine, error):
        engine._sync = lambda path: (_ for _ in ()).throw(error)
        with pytest.raises(TranscriptionError) as exc:
            engine.transcribe(Path("a.wav"), 5.0)
        assert not isinstance(exc.value, TransientError)


class TestBuildEngine:
    def test_mock(self):
        assert isinstance(build_engine(Settings(engine="mock")), MockEngine)

    def test_google_requires_project(self):
        with pytest.raises(TranscriptionError, match="GOOGLE_CLOUD_PROJECT"):
            build_engine(Settings(engine="google", gcp_project=None))

    def test_unknown_engine(self):
        with pytest.raises(ValueError, match="Unknown STT_ENGINE"):
            build_engine(Settings(engine="nope"))


class FakeSpeechClient:
    """Records requests and returns canned results, standing in for SpeechClient."""

    def __init__(self, results=None, batch_error=None):
        self.results = results or []
        self.batch_error = batch_error
        self.requests = []

    def recognize(self, request):
        self.requests.append(request)
        return SimpleNamespace(results=self.results)

    def batch_recognize(self, request):
        self.requests.append(request)
        uri = request.files[0].uri

        def finish(timeout):
            if self.batch_error:
                raise self.batch_error
            return SimpleNamespace(results={uri: SimpleNamespace(
                transcript=SimpleNamespace(results=self.results))})

        return SimpleNamespace(result=finish)


class FakeBlob:
    def __init__(self, name):
        self.name = name
        self.uploaded_from = None
        self.deleted = False

    def upload_from_filename(self, filename):
        self.uploaded_from = filename

    def delete(self):
        self.deleted = True


@pytest.fixture
def google(monkeypatch):
    """Real GoogleSTTEngine wired to fake Speech and Storage clients."""
    import google.cloud.speech_v2
    import google.cloud.storage

    speech = FakeSpeechClient(results=[result("Hello.", end=1.0, words=[word(0.2)])])
    blobs = []

    def make_blob(name):
        blobs.append(FakeBlob(name))
        return blobs[-1]

    monkeypatch.setattr(google.cloud.speech_v2, "SpeechClient", lambda: speech)
    monkeypatch.setattr(google.cloud.storage, "Client",
                        lambda: SimpleNamespace(bucket=lambda name: SimpleNamespace(blob=make_blob)))
    engine = GoogleSTTEngine(project="proj", language="en-GB", bucket="my-bucket")
    return SimpleNamespace(engine=engine, speech=speech, blobs=blobs)


class TestGoogleRequests:
    def test_sync_sends_audio_bytes_with_config(self, google, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"audio-bytes")

        segments = google.engine.transcribe(path, duration=5.0)

        request = google.speech.requests[0]
        assert request.recognizer == "projects/proj/locations/global/recognizers/_"
        assert request.content == b"audio-bytes"
        assert list(request.config.language_codes) == ["en-GB"]
        assert request.config.features.enable_word_time_offsets
        assert [(s.start, s.end, s.text) for s in segments] == [(0.2, 1.0, "Hello.")]

    def test_batch_uploads_to_gcs_and_cleans_up(self, google, tmp_path):
        path = tmp_path / "long.mp3"
        path.write_bytes(b"x")

        segments = google.engine.transcribe(path, duration=600.0)

        blob = google.blobs[0]
        assert blob.uploaded_from == str(path)
        assert blob.name.startswith("stt/") and blob.name.endswith(".mp3")
        assert google.speech.requests[0].files[0].uri == f"gs://my-bucket/{blob.name}"
        assert blob.deleted
        assert segments[0].text == "Hello."

    def test_batch_deletes_upload_even_when_recognition_fails(self, google, tmp_path):
        google.speech.batch_error = gexc.ServiceUnavailable("down")
        path = tmp_path / "long.mp3"
        path.write_bytes(b"x")

        with pytest.raises(TransientError):
            google.engine.transcribe(path, duration=600.0)
        assert google.blobs[0].deleted
