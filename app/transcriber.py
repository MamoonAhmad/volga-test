"""Speech-to-text engines behind one interface."""
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None


class TranscriptionError(Exception):
    """Permanent failure: retrying will not help (bad audio, bad config, auth)."""


class TransientError(TranscriptionError):
    """Temporary failure: safe to retry (timeouts, rate limits, provider 5xx)."""


class SpeechEngine(Protocol):
    def transcribe(self, path: Path, duration: float | None) -> list[Segment]: ...


class GoogleSTTEngine:
    """Google Cloud Speech-to-Text v2.

    Short audio uses synchronous recognize; longer audio is uploaded to GCS and
    sent through batch_recognize, which handles files up to several hours.
    """

    SYNC_LIMIT_SEC = 55  # Google's sync limit is 60s; leave headroom

    def __init__(self, project: str | None, language: str, bucket: str | None):
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech as cs

        if not project:
            raise TranscriptionError("GOOGLE_CLOUD_PROJECT is not set")
        self.cs = cs
        self.client = SpeechClient()
        self.bucket = bucket
        self.recognizer = f"projects/{project}/locations/global/recognizers/_"
        self.config = cs.RecognitionConfig(
            auto_decoding_config=cs.AutoDetectDecodingConfig(),
            language_codes=[language],
            model="long",
            features=cs.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
            ),
        )

    def transcribe(self, path: Path, duration: float | None) -> list[Segment]:
        from google.api_core import exceptions as gexc

        try:
            if duration is not None and duration > self.SYNC_LIMIT_SEC:
                results = self._batch(path)
            else:
                results = self._sync(path)
        except (gexc.ServiceUnavailable, gexc.DeadlineExceeded,
                gexc.ResourceExhausted, gexc.InternalServerError) as e:
            raise TransientError(f"Google STT unavailable: {e.message}") from e
        except gexc.GoogleAPICallError as e:
            raise TranscriptionError(f"Google STT rejected request: {e.message}") from e
        return self._to_segments(results)

    def _sync(self, path: Path):
        request = self.cs.RecognizeRequest(
            recognizer=self.recognizer, config=self.config, content=path.read_bytes()
        )
        return self.client.recognize(request=request).results

    def _batch(self, path: Path):
        if not self.bucket:
            raise TranscriptionError("Audio longer than ~1 minute requires GCS_BUCKET to be set")
        from google.cloud import storage

        blob = storage.Client().bucket(self.bucket).blob(f"stt/{uuid.uuid4().hex}{path.suffix}")
        blob.upload_from_filename(str(path))
        uri = f"gs://{self.bucket}/{blob.name}"
        try:
            operation = self.client.batch_recognize(request=self.cs.BatchRecognizeRequest(
                recognizer=self.recognizer,
                config=self.config,
                files=[self.cs.BatchRecognizeFileMetadata(uri=uri)],
                recognition_output_config=self.cs.RecognitionOutputConfig(
                    inline_response_config=self.cs.InlineOutputConfig()
                ),
            ))
            response = operation.result(timeout=3 * 60 * 60)
            return response.results[uri].transcript.results
        finally:
            blob.delete()

    @staticmethod
    def _to_segments(results) -> list[Segment]:
        segments, prev_end = [], 0.0
        for r in results:
            if not r.alternatives or not r.alternatives[0].transcript.strip():
                continue
            alt = r.alternatives[0]
            start = alt.words[0].start_offset.total_seconds() if alt.words else prev_end
            end = r.result_end_offset.total_seconds()
            segments.append(Segment(round(start, 2), round(end, 2),
                                    alt.transcript.strip(), round(alt.confidence, 3)))
            prev_end = end
        return segments


class MockEngine:
    """Deterministic engine for tests and local development without credentials."""

    def transcribe(self, path: Path, duration: float | None) -> list[Segment]:
        return [
            Segment(0.0, 2.5, "Hello, this is a test recording.", 0.98),
            Segment(2.5, 5.0, "The transcription service is working.", 0.95),
        ]


def build_engine(settings: Settings) -> SpeechEngine:
    if settings.engine == "google":
        return GoogleSTTEngine(settings.gcp_project, settings.language, settings.gcs_bucket)
    if settings.engine == "mock":
        return MockEngine()
    raise ValueError(f"Unknown STT_ENGINE {settings.engine!r}")
