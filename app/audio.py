"""Audio format validation and normalisation."""
import shutil
import subprocess
from pathlib import Path

import mutagen

# Formats the STT engines decode directly.
NATIVE_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}
# Formats converted to 16 kHz mono FLAC with ffmpeg before transcription.
CONVERTIBLE_FORMATS = {".m4a", ".aac", ".mp4", ".webm"}
SUPPORTED_FORMATS = NATIVE_FORMATS | CONVERTIBLE_FORMATS


class InvalidAudio(ValueError):
    pass


def probe_duration(path: Path) -> float | None:
    """Read duration from the file header. Raises InvalidAudio if a native file is unreadable."""
    try:
        info = mutagen.File(path)
    except mutagen.MutagenError as e:
        raise InvalidAudio(f"Unreadable audio file: {e}") from e
    if info is None or info.info is None:
        if path.suffix.lower() in NATIVE_FORMATS:
            raise InvalidAudio("File content is not valid audio")
        return None  # e.g. webm: mutagen can't parse it, ffmpeg will
    return round(info.info.length, 2)


def normalize(path: Path) -> Path:
    """Return a path the engine can read, converting non-native formats to FLAC."""
    if path.suffix.lower() in NATIVE_FORMATS:
        return path
    if shutil.which("ffmpeg") is None:
        raise InvalidAudio(f"{path.suffix} requires ffmpeg, which is not installed")
    out = path.with_suffix(".flac")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", str(out)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InvalidAudio(f"Could not decode audio: {result.stderr.strip()[:200]}")
    return out
