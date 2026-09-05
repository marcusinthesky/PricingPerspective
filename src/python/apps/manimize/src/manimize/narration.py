"""Synthesize NARRATION.md with Kokoro-82M and mux it into scene videos."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import imageio_ffmpeg
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    class KokoroPipeline(Protocol):
        """Callable subset of Kokoro's pipeline used by this module."""

        def __call__(
            self,
            text: str,
            *,
            voice: str,
            speed: float,
        ) -> Iterable[tuple[str, str, np.ndarray]]:
            """Generate one or more audio chunks from text."""
            ...


APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = APP_ROOT / "NARRATION.md"
DEFAULT_SOURCE_DIR = APP_ROOT / "videos"
DEFAULT_AUDIO_DIR = APP_ROOT / "audio"
DEFAULT_OUTPUT_DIR = APP_ROOT / "timed_videos"
SAMPLE_RATE = 24_000
HEADING = re.compile(r"^## (?P<number>\d{2}) — (?P<title>.+?)\s*$")
SCENE_SLUGS = {
    "01": "firm_as_distribution",
    "02": "couplings_and_wasserstein",
    "03": "polarization_to_covariance",
    "04": "covariance_envelope",
    "05": "transmission_and_slack",
    "06": "target_anchored_reconstruction",
    "07": "symmetric_distance_directed_field",
    "08": "quadratic_adjustment_spatial_closure",
    "09": "pairwise_not_jointly_coherent",
    "10": "multifirm_dispersion_barycentre",
    "11": "portfolio_variance_certificate",
    "12": "certificate_convexity",
    "13": "three_optimizations",
}
LOGGER = logging.getLogger(__name__)
FULL_SERIES_NAME = "00_full_series_with_audio.mp4"


@dataclass(frozen=True)
class Narration:
    """One numbered narration section and its normalized prose."""

    number: str
    title: str
    text: str

    @property
    def stem(self) -> str:
        """Return the numbered scene stem used by the reference videos."""
        try:
            slug = SCENE_SLUGS[self.number]
        except KeyError as exc:
            raise ValueError(f"Unknown narration scene number: {self.number}") from exc
        return f"{self.number}_{slug}"


def parse_narration(path: Path) -> list[Narration]:
    """Parse numbered ``##`` sections from a Markdown narration script."""
    sections: list[Narration] = []
    current_number: str | None = None
    current_title = ""
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match:
            if current_number is not None:
                sections.append(_section(current_number, current_title, lines))
            current_number = match.group("number")
            current_title = match.group("title")
            lines = []
        elif current_number is not None:
            lines.append(line)
    if current_number is not None:
        sections.append(_section(current_number, current_title, lines))
    if not sections:
        message = f"No numbered narration sections found in {path}"
        raise ValueError(message)
    numbers = [section.number for section in sections]
    if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
        message = f"Narration section numbers must be unique and ordered: {numbers}"
        raise ValueError(message)
    return sections


def _section(number: str, title: str, lines: list[str]) -> Narration:
    text = " ".join(line.strip() for line in lines if line.strip())
    if not text:
        message = f"Narration section {number} has no prose"
        raise ValueError(message)
    return Narration(number, title, text)


def _write_audio(audio: np.ndarray, destination: Path) -> float:
    sf = importlib.import_module("soundfile")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, SAMPLE_RATE, subtype="PCM_16")
    return len(audio) / SAMPLE_RATE


def synthesize(
    section: Narration,
    destination: Path,
    pipeline: KokoroPipeline,
    voice: str,
    speed: float,
) -> float:
    """Generate one section with the official Kokoro Python pipeline."""
    chunks = [audio for _, _, audio in pipeline(section.text, voice=voice, speed=speed)]
    if not chunks:
        message = f"Kokoro returned no audio for section {section.number}"
        raise RuntimeError(message)
    return _write_audio(np.concatenate(chunks), destination)


def probe_duration(path: Path) -> float:
    """Read a media duration using the pinned ffmpeg binary."""
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-i",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"Duration: (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)",
        result.stderr,
    )
    if match is None:
        message = f"Could not read media duration: {path}"
        raise ValueError(message)
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + float(match.group("seconds"))
    )


def mux_timed_video(
    source: Path,
    audio: Path,
    destination: Path,
    narration_duration: float,
    fit: str,
) -> None:
    """Mux audio and make the video duration exactly equal to the narration."""
    source_duration = probe_duration(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fit == "stretch":
        video_filter = f"setpts=PTS*{narration_duration / source_duration:.12f}"
    else:
        pad = max(0.0, narration_duration - source_duration)
        video_filter = f"tpad=stop_mode=clone:stop_duration={pad:.6f}"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(audio),
            "-filter_complex",
            f"[0:v]{video_filter},format=yuv420p[v]",
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-t",
            f"{narration_duration:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def concatenate_timed_videos(
    paths: Sequence[Path], destination: Path, duration: float | None = None
) -> None:
    """Concatenate narration-timed scene videos into one MP4."""
    if not paths:
        raise ValueError("At least one timed video is required")
    target_duration = duration or sum(probe_duration(path) for path in paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    listing = destination.with_suffix(".concat.txt")
    listing.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in paths),
        encoding="utf-8",
    )
    try:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{target_duration:.6f}",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True)
    finally:
        listing.unlink(missing_ok=True)


def render(
    script: Path,
    source_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    voice: str,
    speed: float,
    fit: str,
) -> None:
    """Render all narration sections and produce narration-timed MP4s."""
    sections = parse_narration(script)
    numbers = [section.number for section in sections]
    expected = list(SCENE_SLUGS)
    if numbers != expected:
        message = f"Narration sections must be exactly {expected}, got {numbers}"
        raise ValueError(message)
    pipeline_type = cast(
        "Callable[..., KokoroPipeline]", importlib.import_module("kokoro").KPipeline
    )
    pipeline = pipeline_type(lang_code=voice[0])
    durations: list[tuple[Narration, float]] = []
    timed_videos: list[Path] = []
    manifest: list[dict[str, str | float]] = []
    for section in sections:
        audio_path = audio_dir / f"{section.stem}.wav"
        LOGGER.info("Synthesizing %s: %s", section.number, section.title)
        duration = synthesize(section, audio_path, pipeline, voice, speed)
        source = source_dir / f"{section.stem}.mp4"
        if not source.exists():
            message = f"Missing source video for section {section.number}: {source}"
            raise FileNotFoundError(message)
        output = output_dir / f"{section.stem}.mp4"
        mux_timed_video(source, audio_path, output, duration, fit)
        durations.append((section, duration))
        timed_videos.append(output)
        manifest.append(
            {
                "number": section.number,
                "title": section.title,
                "voice": voice,
                "speed": speed,
                "fit": fit,
                "audio": audio_path.name,
                "source_video": source.name,
                "timed_video": output.name,
                "narration_duration_seconds": duration,
                "timed_video_duration_seconds": probe_duration(output),
            }
        )
        LOGGER.info("  %.2fs -> %s", duration, output)
    total = sum(duration for _, duration in durations)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_series = output_dir / FULL_SERIES_NAME
    concatenate_timed_videos(timed_videos, full_series, total)
    (output_dir / "timings.json").write_text(
        json.dumps(
            {
                "model": "hexgrad/Kokoro-82M",
                "full_series": full_series.name,
                "scenes": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Rendered %d narration-timed videos (%.2fs total): %s",
        len(durations),
        total,
        full_series,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the narration CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--voice", default="af_heart", help="Kokoro voice (default: af_heart)"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--fit",
        choices=("hold", "stretch"),
        default="hold",
        help=(
            "Keep video speed and hold its final frame, or stretch all frames "
            "(default: hold)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the narration synthesis and video timing pipeline."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render(
        args.script,
        args.source_dir,
        args.audio_dir,
        args.output_dir,
        args.voice,
        args.speed,
        args.fit,
    )
