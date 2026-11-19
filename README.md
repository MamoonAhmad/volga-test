# Transcription Service

A Python (FastAPI) service that accepts an audio file, transcribes it with Google Cloud
Speech-to-Text (or a local Whisper model), and returns the text with timestamps per segment.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/transcriptions` | Upload audio (multipart field `file`). Returns `202` with a job. |
| `GET` | `/v1/transcriptions/{id}` | Job status (`queued`, `processing`, `done`, `failed`) and segments. |
| `GET` | `/health` | Liveness check. |

```bash
curl -F "file=@call.mp3" localhost:8000/v1/transcriptions
# {"id": "3f2a...", "status": "queued", ...}

curl localhost:8000/v1/transcriptions/3f2a...
```

```json
{
  "id": "3f2a...",
  "filename": "call.mp3",
  "status": "done",
  "attempts": 1,
  "duration_sec": 7.2,
  "segments": [
    {"start": 0.1, "end": 3.4, "text": "Hi, I'm calling about my account.", "confidence": 0.94},
    {"start": 3.9, "end": 7.2, "text": "Can you update my email address?", "confidence": 0.91}
  ],
  "error": null
}
```

Errors: `415` unsupported format, `413` too large, `400` empty file, `422` corrupt audio,
`404` unknown job. Interactive docs at `/docs`.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Google STT
export GOOGLE_APPLICATION_CREDENTIALS=service-account.json
export GOOGLE_CLOUD_PROJECT=my-project
export GCS_BUCKET=my-bucket          # only needed for audio longer than ~1 minute
uvicorn app.main:app --port 8000

# Without credentials (returns fixed sample output)
STT_ENGINE=mock uvicorn app.main:app --port 8000
```

Docker: `docker build -t transcription . && docker run -p 8000:8000 -e STT_ENGINE=mock transcription`

Tests: `pip install -r requirements-dev.txt && pytest`

| Variable | Default | Purpose |
|---|---|---|
| `STT_ENGINE` | `google` | `google`, `whisper` (needs `pip install faster-whisper`), or `mock` |
| `STT_LANGUAGE` | `en-US` | Language code |
| `MAX_UPLOAD_MB` | `100` | Upload size limit |
| `MAX_WORKERS` | `4` | Concurrent transcriptions |
| `MAX_RETRIES` | `3` | Attempts for transient failures |

## Design decisions

**Asynchronous jobs.** Transcription can take minutes, so the upload returns `202` with a
job id straight away and the client polls for the result. Long-held HTTP requests time out
and tie up the server.

**Audio formats.** The extension is checked first (`415`), then the file header is parsed so
corrupt files are rejected at upload (`422`) rather than failing later. WAV, MP3, FLAC and OGG
go to the engine unchanged (Google auto-detects encoding). M4A, AAC and WebM are converted
with ffmpeg to 16 kHz mono FLAC.

**Long audio.** Google's synchronous API accepts about a minute. Longer files are uploaded to
GCS and sent through `batch_recognize`, chosen automatically from the file's duration. The
Whisper engine has no length limit since it processes 30-second windows internally.

**Concurrency.** Uploads are streamed to disk in 1 MB chunks, so large files never sit fully
in memory. Transcription runs in a bounded worker pool (`MAX_WORKERS`), so bursts of uploads
queue up instead of overloading the server or the provider's rate limit. Identical files are
detected by SHA-256 and reuse the existing job.

**Retries.** Errors are split into transient (timeouts, rate limits, provider 5xx), which are
retried with exponential backoff and jitter, and permanent (bad audio, bad credentials), which
fail immediately. A failed job can be resubmitted by uploading the same file again.

**Pluggable engines.** Google STT, local Whisper and a mock all implement one
`SpeechEngine` interface. Cloud APIs need no infrastructure but cost per minute and send audio
to a third party; Whisper is free per minute and keeps data private but needs a GPU for speed.
The mock lets the whole pipeline be tested without credentials.

## Production changes

This keeps job state in memory and runs workers in-process to stay small. For production:

- Job store in Postgres or Redis, and a queue (Celery, SQS, Pub/Sub) with separate worker
  processes, so jobs survive restarts and workers scale independently of the API.
- Uploads in object storage (S3/GCS) instead of local disk, with a retention policy.
- Dead-letter queue and alerting for jobs that exhaust retries; optional fallback engine.
- API keys, per-client rate limits, and webhook callbacks instead of polling.
