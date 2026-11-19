import shutil
import subprocess

import pytest

from app import audio
from app.audio import InvalidAudio, normalize, probe_duration
from tests.conftest import make_wav

has_ffmpeg = shutil.which("ffmpeg") is not None


def test_probe_duration_reads_wav_header(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(make_wav(seconds=2.5))
    assert probe_duration(path) == 2.5


def test_probe_rejects_corrupt_native_file(tmp_path):
    path = tmp_path / "a.mp3"
    path.write_bytes(b"\x00garbage" * 50)
    with pytest.raises(InvalidAudio):
        probe_duration(path)


def test_probe_returns_none_for_unparseable_convertible_format(tmp_path):
    path = tmp_path / "a.webm"
    path.write_bytes(b"\x00garbage" * 50)
    assert probe_duration(path) is None  # left for ffmpeg to decide


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".flac", ".ogg"])
def test_native_formats_are_not_converted(tmp_path, ext):
    path = tmp_path / f"a{ext}"
    path.write_bytes(b"x")
    assert normalize(path) == path


def test_conversion_without_ffmpeg_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _: None)
    path = tmp_path / "a.m4a"
    path.write_bytes(b"x")
    with pytest.raises(InvalidAudio, match="requires ffmpeg"):
        normalize(path)


@pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg not installed")
def test_converts_m4a_to_16k_mono_flac(tmp_path):
    wav = tmp_path / "src.wav"
    wav.write_bytes(make_wav(seconds=1.0, rate=44100))
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), str(m4a)], check=True)

    out = normalize(m4a)

    assert out.suffix == ".flac"
    assert probe_duration(out) == pytest.approx(1.0, abs=0.1)


@pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg not installed")
def test_corrupt_convertible_file_fails_clearly(tmp_path):
    path = tmp_path / "a.m4a"
    path.write_bytes(b"\x00garbage" * 50)
    with pytest.raises(InvalidAudio, match="Could not decode"):
        normalize(path)
