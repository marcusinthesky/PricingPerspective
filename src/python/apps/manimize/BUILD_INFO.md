# Build information

## Native source target

- **Library:** Manim Community Edition
- **Target version:** 0.21.0
- **Renderer:** Cairo-compatible 2D scene code
- **Default project canvas:** 1280×720, 16:9
- **Default native frame rate:** 30 fps via `manim.cfg`
- **Mathematical typesetting:** `MathTex` / LaTeX

## Included MP4 files

The MP4s in `videos/` are deterministic **reference renders**, not native Manim outputs.

They were created in the artifact-building environment with the included `tools/preview_renderer.py` because the environment did not have Manim installed and external Python package downloads were unavailable. The renderer mirrors the scene composition with Pillow, Matplotlib mathtext, and ffmpeg.

- Resolution: 1280×720
- Frame rate: 20 fps
- Codec: H.264 / yuv420p
- Audio: none
- Duration: 7.5 seconds per scene
- Combined duration: 97.5 seconds

The authoritative editable sources are the thirteen files in `scenes/`. Re-render them with `uv run --project . --extra native manimize-render`.

`NARRATION.md` is synthesized locally with the official Kokoro Python pipeline
using the `hexgrad/Kokoro-82M` weights. `manimize-narrate` writes 24 kHz WAV
files and muxes them into duration-matched MP4s under `timed_videos/` without
modifying the silent reference videos. It also concatenates those scene files
into `timed_videos/00_full_series_with_audio.mp4`.

The portfolio Animated companion is sourced from the narration-timed series,
not from the silent reference MP4. Its site exports preserve the generated
audio track in both MP4 and WebM containers.

## Validation performed

- All project Python files compile under Python 3.13.
- Every included MP4 was parsed with `ffprobe` and checked for the expected dimensions, duration, and video stream.
- Final frames were reviewed together in `thumbnails/contact_sheet.png`.
- Scene 09's three pairwise permutations were verified against exhaustive squared-Euclidean assignment costs for the documented synthetic point clouds.
