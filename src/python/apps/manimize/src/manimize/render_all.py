#!/usr/bin/env python3
"""Render the manuscript animation series with native ManimCE.

Example:
    python render_all.py --quality m
    python render_all.py --quality h --scene 03 --scene 04 --no-concat
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT / "scenes"
MEDIA_DIR = ROOT / "native_media"
OUTPUT_DIR = ROOT / "native_renders"


@dataclass(frozen=True)
class SceneSpec:
    number: str
    slug: str
    class_name: str

    @property
    def source(self) -> Path:
        return SCENE_DIR / f"{self.number}_{self.slug}.py"

    @property
    def output_name(self) -> str:
        return f"{self.number}_{self.slug}.mp4"


SCENES = [
    SceneSpec("01", "firm_as_distribution", "FirmAsDistribution"),
    SceneSpec("02", "couplings_and_wasserstein", "CouplingsAndWasserstein"),
    SceneSpec("03", "polarization_to_covariance", "PolarizationToCovariance"),
    SceneSpec("04", "covariance_envelope", "CovarianceEnvelope"),
    SceneSpec("05", "transmission_and_slack", "TransmissionAndSlack"),
    SceneSpec("06", "target_anchored_reconstruction", "TargetAnchoredReconstruction"),
    SceneSpec(
        "07", "symmetric_distance_directed_field", "SymmetricDistanceDirectedField"
    ),
    SceneSpec(
        "08",
        "quadratic_adjustment_spatial_closure",
        "QuadraticAdjustmentSpatialClosure",
    ),
    SceneSpec("09", "pairwise_not_jointly_coherent", "PairwiseNotJointlyCoherent"),
    SceneSpec("10", "multifirm_dispersion_barycentre", "MultifirmDispersionBarycentre"),
    SceneSpec("11", "portfolio_variance_certificate", "PortfolioVarianceCertificate"),
    SceneSpec("12", "certificate_convexity", "CertificateConvexity"),
    SceneSpec("13", "three_optimizations", "ThreeOptimizations"),
]

QUALITY_FLAGS = {
    "l": "-ql",
    "m": "-qm",
    "h": "-qh",
    "p": "-qp",
    "k": "-qk",
}


def check_prerequisites(concat: bool) -> None:
    if importlib.util.find_spec("manim") is None:
        raise SystemExit(
            "Manim is not installed in this Python environment. "
            "Run: uv run --project . --extra native manimize-render"
        )
    if concat:
        try:
            imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError as exc:
            raise SystemExit(
                "An ffmpeg executable is required to concatenate the rendered scenes."
            ) from exc


def render_scene(spec: SceneSpec, quality: str, overwrite: bool) -> Path:
    if not spec.source.exists():
        raise FileNotFoundError(f"Missing source file: {spec.source}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / spec.output_name
    if destination.exists() and not overwrite:
        print(f"Skipping existing {destination.name}")
        return destination

    started = time.time()
    command = [
        sys.executable,
        "-m",
        "manim",
        QUALITY_FLAGS[quality],
        "--renderer=cairo",
        "--format=mp4",
        "--media_dir",
        str(MEDIA_DIR),
        "--output_file",
        spec.output_name,
        str(spec.source),
        spec.class_name,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
    )
    print(f"Rendering {spec.number}: {spec.class_name}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)

    candidates = [
        path
        for path in MEDIA_DIR.rglob(spec.output_name)
        if path.is_file() and path.stat().st_mtime >= started - 2
    ]
    if not candidates:
        # Permit cached/native renders whose timestamp did not change.
        candidates = [
            path for path in MEDIA_DIR.rglob(spec.output_name) if path.is_file()
        ]
    if not candidates:
        message = (
            f"Manim completed but {spec.output_name} could not be located in "
            f"{MEDIA_DIR}"
        )
        raise RuntimeError(message)

    rendered = max(candidates, key=lambda path: path.stat().st_mtime)
    shutil.copy2(rendered, destination)
    return destination


def concatenate(paths: list[Path]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    listing = OUTPUT_DIR / "concat.txt"
    listing.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in paths),
        encoding="utf-8",
    )
    output = OUTPUT_DIR / "00_full_series.mp4"
    subprocess.run(
        [
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
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    listing.unlink(missing_ok=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        choices=QUALITY_FLAGS,
        default="m",
        help="Manim quality: l, m, h, p, or k (default: m)",
    )
    parser.add_argument(
        "--scene",
        action="append",
        choices=[spec.number for spec in SCENES],
        help="Render only this scene number; repeat to select several scenes.",
    )
    parser.add_argument(
        "--no-concat", action="store_true", help="Do not build the combined MP4."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files already in native_renders/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [spec for spec in SCENES if not args.scene or spec.number in args.scene]
    do_concat = not args.no_concat and len(selected) == len(SCENES)
    check_prerequisites(do_concat)

    rendered = [render_scene(spec, args.quality, args.overwrite) for spec in selected]
    if do_concat:
        combined = concatenate(rendered)
        print(f"Combined series: {combined}")
    print(f"Rendered files: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
