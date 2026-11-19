# Transcription Service

A Python (FastAPI) service that accepts an audio file, transcribes it with Google Cloud
Speech-to-Text, and returns the text with timestamps per segment.

## Quick start

```bash
# No credentials needed: runs with the mock engine, which returns fixed sample output
docker compose up --build
curl -F "file=@sample.wav" localhost:8000/v1/transcriptions
```

Interactive API docs: http://localhost:8000/docs

### Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then fill in your Google Cloud values
uvicorn app.main:app --port 8000
```

### Tests and lint

```bash
pytest            # runs with coverage; fails below 90%
ruff check .
```

CI (GitHub Actions) runs both on every push, with ffmpeg installed so the format conversion
tests run too.

## API

**`POST /v1/transcriptions`**: upload audio as multipart field `file`. Returns `202 Accepted`
with a job; transcription runs in the background.

```bash
curl -F "file=@call.mp3" localhost:8000/v1/transcriptions
```

**`GET /v1/transcriptions/{id}`**: job status (`queued`, `processing`, `done`, `failed`) and,
when done, the timestamped segments.

```json
{
  "id": "3f2a9c...",
  "filename": "call.mp3",
  "status": "done",
  "attempts": 1,
  "duration_sec": 7.2,
  "segments": [
    {"start": 0.1, "end": 3.4, "text": "Hi, I'm calling about my account.", "confidence": 0.94},
    {"start": 3.9, "end": 7.2, "text": "Can you update my email address?", "confidence": 0.91}
  ],
  "error": null,
  "created_at": "2026-11-19T10:00:00Z",
  "updated_at": "2026-11-19T10:00:04Z"
}
```

**`GET /health`**: liveness check (no auth).

Error responses: `400` empty file, `401` missing or invalid API key, `404` unknown job,
`413` file too large, `415` unsupported format, `422` corrupt audio.

If `API_KEY` is set, send it as the `X-API-Key` header on `/v1` endpoints.

## Configuration

All settings come from environment variables or a `.env` file; see
[`.env.example`](.env.example) for the full list with descriptions. The main ones:

- `STT_ENGINE`: `google` (default) or `mock`
- `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`: Google Cloud credentials and project
- `GCS_BUCKET`: needed only for audio longer than about a minute
- `API_KEY`: enables authentication when set
- `MAX_UPLOAD_MB`, `MAX_WORKERS`, `MAX_RETRIES`: limits

## Project layout

```
app/
  main.py         API routes, upload streaming, auth
  jobs.py         job tracking, worker pool, retries
  transcriber.py  SpeechEngine interface, Google STT engine, mock engine
  audio.py        format validation and conversion
  config.py       settings from environment
tests/            API, engine and audio tests (fake Google clients, no network)
```

## Design decisions

### Accepting audio files
Uploads are streamed to disk in 1 MB chunks, so a large file never sits fully in memory, and
are hashed on the way in. The extension is checked first (`415`), then the file header is
parsed so corrupt or mislabelled files are rejected at upload (`422`) instead of failing later
in a worker. Audio files are deleted once their job finishes; only the transcript is kept.

### Handling different audio formats
WAV, MP3, FLAC and OGG go to Google unchanged; its `AutoDetectDecodingConfig` reads encoding
and sample rate from the header, so nothing is hard-coded per format. M4A, AAC, MP4 and WebM
are converted with ffmpeg to 16 kHz mono FLAC, which is lossless and matches what speech models
expect. Every input therefore reaches the engine in a format it reads.

### Timestamps per segment
Word time offsets are enabled. Each segment starts at its first word's offset and ends at
Google's `result_end_offset`. If a segment has no word offsets, it starts where the previous
one ended, so timestamps never go backwards.

### Long audio files
Google's synchronous API accepts about one minute. The service reads the file's duration and
routes automatically: short files use `recognize`; longer files are uploaded to GCS and sent
through `batch_recognize`, which handles several hours per job. The temporary GCS object is
deleted afterwards, including when recognition fails.

### Concurrent uploads
Uploads and transcription are decoupled. The upload endpoint only saves and validates the file,
then returns a job id. Transcription runs in a bounded worker pool (`MAX_WORKERS`), so bursts of
uploads queue up instead of overloading the server or the provider's rate limit. Identical
files are detected by SHA-256 and reuse the existing job rather than paying to transcribe twice.

### Retries and recovery
Errors are classified. Transient errors (timeouts, rate limits, Google 5xx) are retried with
exponential backoff plus jitter, so concurrent retries don't all hit the provider at once.
Permanent errors (bad audio, invalid credentials, missing configuration) fail immediately,
because retrying cannot fix them. Unexpected exceptions are logged with a stack trace and the job
is marked failed with a generic message, so internal details are not exposed to clients. Every
job records its attempt count and last error, and a failed job can be resubmitted by uploading
the same file again.

### Exposing it as an API
Asynchronous REST: `202` plus a job id on upload, then the client polls for the result. Long
transcriptions would time out a synchronous HTTP request and tie up the server. Paths are
versioned (`/v1`), responses have explicit schemas (visible at `/docs`), and status codes tell
the client whether the problem is theirs (`4xx`) or not.

## Other speech-to-text options

The engine sits behind a small interface, so the provider is a configuration choice rather than
a rewrite:

```python
class SpeechEngine(Protocol):
    def transcribe(self, path: Path, duration: float | None) -> list[Segment]: ...
```

Alternatives that fit this interface:

- **Local open-source models** such as Whisper, faster-whisper or WhisperX: no per-minute cost
  and audio never leaves your infrastructure, which matters for sensitive recordings. Needs a
  GPU for good speed, and you run the model yourself. Whisper processes 30-second windows
  internally, so long files need no special handling.
- **Other cloud APIs** such as Deepgram, AWS Transcribe or Azure Speech: no infrastructure to
  run, and each has its own batch or callback mode for long files.

A generic example of adding an engine, here a local Whisper model:

```python
# app/transcriber.py
class WhisperEngine:
    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel  # pip install faster-whisper
        self.model = WhisperModel(model_size, device="auto", compute_type="int8")

    def transcribe(self, path: Path, duration: float | None) -> list[Segment]:
        segments, _ = self.model.transcribe(str(path), vad_filter=True)
        return [Segment(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segments]


def build_engine(settings: Settings) -> SpeechEngine:
    ...
    if settings.engine == "whisper":
        return WhisperEngine()
```

Then run with `STT_ENGINE=whisper`. Nothing else changes: the API, job handling, retries and
response format stay the same. To make its errors retryable, raise `TransientError` for
temporary failures and `TranscriptionError` for permanent ones, as the Google engine does.

## Limitations and production changes

To keep it simple to run and review, this version keeps job state in memory and runs workers
inside the API process. For production I would change:

- **Durable jobs:** store jobs in Postgres or Redis and use a queue (Celery, SQS or Pub/Sub)
  with separate worker processes, so jobs survive restarts and workers scale independently.
- **Storage:** keep uploads in object storage (S3 or GCS) with a retention policy, not local disk.
- **Long jobs:** a batch job currently occupies a worker thread while Google processes it. With
  a queue, the worker would submit the operation and a separate step would collect the result.
- **Failure handling:** a dead-letter queue and alerts for jobs that use up their retries, and
  optionally a fallback engine.
- **Clients:** per-client API keys and rate limits, and webhook callbacks instead of polling.
- **Observability:** structured logs, metrics (queue depth, latency, error rate) and tracing.
