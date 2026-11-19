# Transcription Service

A Python (FastAPI) service that accepts an audio file, transcribes it with Google Cloud
Speech-to-Text, and returns the text with timestamps per segment.

**Contents:** [Approach](#approach) ·
[Design decisions](#key-design-and-technical-decisions) ·
[Code structure](#code-structure-and-implementation) ·
[Assumptions](#assumptions) · [Trade-offs](#trade-offs) ·
[Limitations and next steps](#known-limitations-and-next-steps) ·
[Running it](#running-it) · [API](#api-reference)

---

## Approach

The problem is to turn an audio file into text with timestamps and make the result easy for
other systems to use. The model itself is solved: a managed speech-to-text API does that well.
The engineering is in everything around it, so that is where I spent the effort.

The starting point was one observation: **transcription is slow and unreliable compared with
a normal web request.** A long recording can take minutes, and a third-party API can time out,
rate-limit or fail. That led to the main design choice:

1. **Separate accepting the file from transcribing it.** The upload endpoint only saves and
   validates the file, then immediately returns a job id (`202 Accepted`).
2. **Transcribe in background workers**, with a fixed limit on how many run at once and
   automatic retries for temporary failures.
3. **Let the client fetch the result** from a status endpoint once the job is done.

The rest follows from that: validate early so bad input fails fast, keep the speech provider
behind a small interface so it can be swapped, and return one structured JSON format that
downstream systems can rely on whatever the provider.

I chose **Google Cloud Speech-to-Text v2** as the engine. It's managed, scales without running
GPUs, returns per-word timing for timestamps, and has a batch mode for long audio. A **mock
engine** with the same interface lets the whole service run and be tested without credentials.

---

## Key design and technical decisions

### 1. Asynchronous job API instead of a synchronous endpoint
**Decision:** `POST` returns a job id at once; the client polls `GET /v1/transcriptions/{id}`.
**Why:** a synchronous request would hold an HTTP connection open for minutes, hit proxy and
client timeouts, and tie up server capacity. Jobs also give a natural place to record status,
attempts and errors.

### 2. Validate at upload, before any expensive work
**Decision:** check the extension (`415`), size (`413`), emptiness (`400`) and, for formats
Google reads directly, parse the file header (`422`) inside the upload request.
**Why:** a bad file should fail immediately with a clear error, not minutes later in a worker
after paying for an API call. Uploaded files are saved under a random UUID, never the
client's filename, so a filename can't be used for path traversal.

### 3. Audio formats: pass through or convert
**Decision:** WAV, MP3, FLAC and OGG are sent to Google unchanged, using its
`AutoDetectDecodingConfig`, which reads encoding and sample rate from the file header.
M4A, AAC, MP4 and WebM are converted with ffmpeg to 16 kHz mono FLAC first.
**Why:** avoids an unnecessary conversion step for common formats, and 16 kHz mono FLAC is
lossless, compact and what speech models are trained on.
*(See [limitations](#known-limitations-and-next-steps): Google's auto-detection covers 16-bit
PCM WAV and Ogg Opus, not every variant, and the next step is to always convert.)*

### 4. Timestamps per segment from Google's own segmentation
**Decision:** request per-word time offsets (`enable_word_time_offsets`). Each result Google
returns becomes one segment: it starts at its first word's offset and ends at the result's
`result_end_offset`. If a result has no word timings, it starts where the previous one ended.
**Why:** Google already splits speech at natural pauses, so there's no need to re-segment the
text. The fallback guarantees timestamps never go backwards.

### 5. Long audio: route by duration
**Decision:** read the duration from the file header. Up to 55 seconds goes to the synchronous
`recognize` call; longer files are uploaded to Cloud Storage and sent to `batch_recognize`,
which handles hours of audio. The temporary Cloud Storage object is always deleted afterwards.
**Why:** Google's synchronous call only accepts about one minute of audio. Choosing the path
automatically means clients don't need to know about the limit.

### 6. Concurrency: a bounded worker pool and deduplication
**Decision:** transcription runs in a thread pool capped at `MAX_WORKERS` (default 4). Each
upload is hashed (SHA-256); uploading a file identical to one already queued, running or done
returns the existing job.
**Why:** a burst of uploads queues up instead of overloading the server or exceeding Google's
rate limits, and the same recording is never paid for twice. Job state is protected by a lock
so concurrent uploads can't create duplicate jobs.

### 7. Retries: classify errors before retrying
**Decision:** errors are split into two types.
- **Transient** (Google unavailable, deadline exceeded, quota exhausted, internal error):
  retried up to `MAX_RETRIES` times with exponential backoff plus random jitter.
- **Permanent** (invalid audio, bad credentials, missing configuration): fail immediately.

**Why:** retrying a bad API key or corrupt file only wastes time and money. Jitter stops many
failed jobs from retrying at the same moment and hitting the provider together. Each job records
its attempt count and last error, and a failed job can be resubmitted by uploading the file
again (failed jobs are excluded from deduplication).

### 8. Pluggable speech engine
**Decision:** the engine is a small interface, `transcribe(path, duration) -> list[Segment]`,
selected by `STT_ENGINE`.
**Why:** the provider becomes a configuration choice. It also makes the service testable:
tests use a mock engine or fake Google clients and need no network.

### 9. Privacy: keep the transcript, not the audio
**Decision:** the uploaded audio (and any converted copy) is deleted when its job finishes,
whether it succeeded or failed. Optional API key authentication protects the endpoints.
**Why:** recordings often contain personal information. Keeping only what downstream systems
need reduces risk.

---

## Code structure and implementation

```
app/
  main.py          API routes, upload handling, validation, API key auth, response schemas
  jobs.py          JobManager: job state, worker pool, deduplication, retries, cleanup
  transcriber.py   SpeechEngine interface, GoogleSTTEngine, MockEngine, error types
  audio.py         format checks, duration probing, ffmpeg conversion
  config.py        settings from environment variables / .env
tests/
  test_api.py          end-to-end API behaviour: validation, retries, dedup, auth, cleanup
  test_transcriber.py  Google response mapping, sync/batch routing, error classification
  test_audio.py        duration probing and format conversion
```

### How a request flows through the code

1. **Upload** (`main.py`, `create_transcription`). The extension is checked against the
   supported formats. `save_upload` writes the file to `uploads/` under a random name in 1 MB
   chunks, computing its SHA-256 and enforcing the size limit as it goes. `probe_duration`
   (`audio.py`) parses the header to reject corrupt files.
2. **Queue** (`jobs.py`, `JobManager.submit`). Under a lock, the hash is checked against
   existing jobs. If it's new, a `Job` is created with status `queued` and handed to the
   thread pool. The endpoint returns the job with `202`.
3. **Prepare** (`jobs.py`, `_run`). A worker marks the job `processing` and calls `normalize`
   (`audio.py`), which converts the file with ffmpeg if Google can't read it directly.
4. **Transcribe** (`jobs.py`, `_transcribe`, then `transcriber.py`). The duration decides the
   path: `GoogleSTTEngine._sync` for short audio, `_batch` (upload to Cloud Storage, then
   `batch_recognize`) for long audio. Google exceptions are mapped to `TransientError` or
   `TranscriptionError`; transient ones go back round the retry loop with backoff.
5. **Build segments** (`transcriber.py`, `_to_segments`). Google's results become `Segment`
   objects with `start`, `end`, `text` and `confidence`.
6. **Finish** (`jobs.py`). The job is marked `done` with its segments, or `failed` with the
   reason. In a `finally` block, the audio files are deleted.
7. **Fetch** (`main.py`, `get_transcription`). The client polls by id and receives the job as
   JSON, validated against the `JobOut` schema.

### Testing and tooling
- 49 tests covering the API end to end, retries, deduplication, authentication, file cleanup,
  Google response mapping, sync/batch routing, error classification and format conversion.
  Coverage is enforced at 90% or above.
- Google is tested with fake clients, so tests run offline in under a second.
- GitHub Actions runs `ruff` and the tests on every push, with ffmpeg installed.
- Docker image runs as a non-root user; `docker compose up` starts the service with the mock
  engine so it can be tried without credentials.

---

## Assumptions

- **Input:** single-speaker or mixed speech in one language per file (default `en-US`,
  configurable), recorded in common formats (WAV, MP3, FLAC, OGG, M4A, AAC, MP4, WebM).
- **Size and length:** files up to 100 MB (configurable). Most are under a minute, but long
  recordings must work too, via batch recognition.
- **Clients** can poll for results; real-time or streaming transcription isn't required.
- **Deployment** for this exercise is a single instance. In-memory job state is acceptable;
  durability across restarts is out of scope (see next steps).
- **Downstream use** means a stable JSON structure with timestamps and confidence that other
  systems can index, display as captions or analyse.
- **Google Cloud** credentials and, for long audio, a Cloud Storage bucket are available.
  Without them, the mock engine shows the full flow.

---

## Trade-offs

- **Managed API vs local model.** Google needs no GPUs or model operations and scales on its
  own, but costs per minute and sends audio to a third party. A local model such as Whisper is
  free per minute and keeps audio private, but needs GPUs for good speed and must be run and
  scaled in-house. The engine interface keeps this a configuration decision
  (see [other options](#other-speech-to-text-options)).
- **In-process thread pool vs a real queue.** A thread pool keeps the service a single process
  that's easy to run and review. A queue (Celery, SQS, Pub/Sub) with separate workers would
  survive restarts and scale independently, at the cost of more infrastructure.
- **Polling vs webhooks.** Polling is simple for any client. Webhooks avoid repeated requests
  but need clients to expose an endpoint and the server to handle delivery retries.
- **Passing formats through vs always converting.** Passing supported formats straight to
  Google avoids a conversion step, but relies on Google's auto-detection handling every variant.
  Always converting is more predictable at the cost of some CPU; I now think that's the better
  choice (see below).
- **Deduplication by content hash.** Saves cost and makes retries by clients safe, but two
  users uploading the same file share one job.
- **Threads vs async.** Google's client library blocks, so threads are the simple, correct
  choice here. The API layer itself is async.

---

## Known limitations and next steps

Things I found when reviewing my own code, roughly in priority order:

1. **Always convert audio before sending it to Google.** Google's auto-detection supports 16-bit
   PCM WAV and Ogg Opus, so 24-bit WAV and Ogg Vorbis files can be rejected. For stereo files
   only the first channel is transcribed, so one side of a stereo call recording can be lost.
   Converting every input to 16 kHz mono FLAC fixes all of these, gives accurate durations,
   and keeps short files well under Google's limits.
2. **Route by size as well as duration.** Google's synchronous call also has a 10 MB limit on
   inline audio. A high-quality stereo WAV under a minute can exceed it and fail. Step 1 largely
   solves this too.
3. **Batch results:** read results from `inline_result` (the current field) and check each
   file's `error` field, so a failed file can't be reported as `done` with no segments.
4. **Retry classification:** also treat HTTP 429/502/504, connection errors and operation
   timeouts as transient.
5. **Upload limits:** FastAPI's form parser receives the whole upload into a temporary file
   before the endpoint runs, so the size limit stops the copy into `uploads/` but not the
   network transfer. In production, enforce the limit at the reverse proxy or from the
   `Content-Length` header, and limit how many jobs can wait in the queue (returning `429`).
6. **Error messages:** errors from Google and ffmpeg are passed to the client as-is. They
   should be mapped to user-friendly messages and the details logged instead.
7. **Corrupt-file checks at upload** apply to formats Google reads directly; converted formats
   are only validated when ffmpeg runs in the worker.

### For production
- **Durable jobs:** jobs in Postgres or Redis, and a queue with separate worker processes, so
  jobs survive restarts and workers scale with queue depth.
- **Storage:** uploads in object storage (S3 or GCS) with a retention policy.
- **Long jobs:** a batch job currently occupies a worker thread while Google processes it. With
  a queue, one step would submit the operation and another would collect the result.
- **Failure handling:** a dead-letter queue and alerts for jobs that use up their retries, and
  optionally a fallback engine.
- **Clients:** per-client API keys and rate limits, and webhook callbacks instead of polling.
- **Observability:** structured logs, metrics (queue depth, latency, error rate) and tracing.
- **Richer output:** a full-text field, word-level timestamps and subtitle formats (SRT/VTT).

---

## Other speech-to-text options

The engine sits behind a small interface, so the provider can change without touching the API,
jobs or retries:

```python
class SpeechEngine(Protocol):
    def transcribe(self, path: Path, duration: float | None) -> list[Segment]: ...
```

- **Local open-source models** such as Whisper, faster-whisper or WhisperX: no per-minute cost,
  and audio never leaves your infrastructure. Needs a GPU for good speed. Whisper processes
  30-second windows internally, so long files need no special handling.
- **Other cloud APIs** such as Deepgram, AWS Transcribe or Azure Speech: no infrastructure to
  run, each with its own batch or callback mode for long files.

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

Then run with `STT_ENGINE=whisper`. To make its errors retryable, raise `TransientError` for
temporary failures and `TranscriptionError` for permanent ones, as the Google engine does.

---

## Running it

### Quick start (no credentials needed)

```bash
docker compose up --build          # uses the mock engine, which returns fixed sample output
curl -F "file=@/path/to/any.wav" localhost:8000/v1/transcriptions
```

Interactive API docs: http://localhost:8000/docs

### Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env               # then fill in your Google Cloud values
uvicorn app.main:app --port 8000
```

### Tests and lint

```bash
pytest            # runs with coverage; fails below 90%
ruff check .
```

### Configuration

All settings come from environment variables or a `.env` file; see
[`.env.example`](.env.example) for the full list. The main ones:

- `STT_ENGINE`: `google` (default) or `mock`
- `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`: Google Cloud credentials and project
- `GCS_BUCKET`: needed only for audio longer than about a minute
- `API_KEY`: enables authentication when set
- `MAX_UPLOAD_MB`, `MAX_WORKERS`, `MAX_RETRIES`: limits

---

## API reference

**`POST /v1/transcriptions`**: upload audio as multipart field `file`. Returns `202 Accepted`
with a job.

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

Errors: `400` empty file, `401` missing or invalid API key, `404` unknown job,
`413` file too large, `415` unsupported format, `422` corrupt audio.
If `API_KEY` is set, send it as the `X-API-Key` header on `/v1` endpoints.
