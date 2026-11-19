import io
import os
import wave

import pytest

# Must be set before app.config is imported so the module-level app uses the mock engine.
os.environ.setdefault("STT_ENGINE", "mock")


def make_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav()
