# Distributional Information Geometry for Financial Dependence — Manim Series

A thirteen-scene animation package that progressively explains the core mathematical ideas of **Distributional Information Geometry for Financial Dependence**.

The sequence follows the manuscripts' conceptual progression:

1. **Pair / separation** — probability laws, couplings, Wasserstein distance, polarization, and covariance envelopes.
2. **Field / reconstruction** — target-anchored barycentric reconstruction, directed interaction fields, and spatial exposure adjustment.
3. **Portfolio / dispersion** — joint-coupling compatibility, multi-firm transport dispersion, portfolio-risk certificates, and convex allocation.

The package targets readers of *Quantitative Finance*, *Journal of Financial Econometrics*, and *European Journal of Finance*. It emphasizes the mathematical transitions that are easiest to misunderstand in static notation.

## Package contents

- `scenes/` — thirteen standalone **Manim Community Edition 0.21.x** scene files.
- `palette.py` and `theme.py` — the portfolio's neutral palette plus shared Manim colors, typography, panels, labels, and mathematical helpers.
- `videos/` — thirteen silent 1280×720 H.264 reference MP4s and one concatenated 97.5-second series.
- `thumbnails/` — a final-frame thumbnail for each scene plus a contact sheet.
- `src/manimize/render_all.py` — native Manim batch renderer and optional ffmpeg concatenation.
- `tools/preview_renderer.py` — deterministic vector fallback used to produce the included reference videos.
- `STORYBOARD.md` — animation beats and pedagogical intent.
- `AUDIENCE_GUIDE.md` — recommended subsets by audience.
- `NARRATION.md` — voice-over copy consumed by the Kokoro-82M pipeline.
- `src/manimize/narration.py` — Kokoro synthesis, duration measurement, video muxing, and full-series concatenation.
- `SOURCE_NOTES.md` — manuscript sections, equations, and theorem anchors.
- `BUILD_INFO.md` — exact distinction between native Manim source and included reference renders.
- `MANIFEST.md` and `SHA256SUMS.txt` — file inventory and integrity hashes.

## Scene index

| No. | Manim class | Source file | Mathematical focus |
|---:|---|---|---|
| 01 | `FirmAsDistribution` | `scenes/01_firm_as_distribution.py` | A firm as a probability law rather than a point |
| 02 | `CouplingsAndWasserstein` | `scenes/02_couplings_and_wasserstein.py` | Couplings and least squared displacement |
| 03 | `PolarizationToCovariance` | `scenes/03_polarization_to_covariance.py` | Polarization connects distance to inner products |
| 04 | `CovarianceEnvelope` | `scenes/04_covariance_envelope.py` | Covariance ceiling, reflected floor, transport excess |
| 05 | `TransmissionAndSlack` | `scenes/05_transmission_and_slack.py` | Bi-Lipschitz / antilipschitz transmission and slack |
| 06 | `TargetAnchoredReconstruction` | `scenes/06_target_anchored_reconstruction.py` | Pairwise alignment followed by a joint simplex problem |
| 07 | `SymmetricDistanceDirectedField` | `scenes/07_symmetric_distance_directed_field.py` | Why symmetric distance can produce a directed field |
| 08 | `QuadraticAdjustmentSpatialClosure` | `scenes/08_quadratic_adjustment_spatial_closure.py` | A quadratic objective derives the spatial autoregression |
| 09 | `PairwiseNotJointlyCoherent` | `scenes/09_pairwise_not_jointly_coherent.py` | Pairwise-optimal plans need not form one joint law |
| 10 | `MultifirmDispersionBarycentre` | `scenes/10_multifirm_dispersion_barycentre.py` | Multi-firm dispersion and its free-centre representation |
| 11 | `PortfolioVarianceCertificate` | `scenes/11_portfolio_variance_certificate.py` | Perfect alignment minus certified dispersion |
| 12 | `CertificateConvexity` | `scenes/12_certificate_convexity.py` | Conditional negative definiteness and convexity |
| 13 | `ThreeOptimizations` | `scenes/13_three_optimizations.py` | A fixed-versus-varying audit across all three essays |

## Native Manim rendering

Python 3.11 or newer is required by ManimCE 0.21.0. A system installation of ffmpeg and LaTeX is also recommended because the scenes use `MathTex`.

```bash
uv sync --project .
uv run --project . --extra native manimize-render --quality m
```

Native outputs are copied to `native_renders/`. The quality choices are:

```text
l  low preview
m  720p30, default
h  1080p60
p  1440p60
k  2160p60
```

Render one scene directly:

```bash
uv run --project . --extra native python -m manim -qm scenes/03_polarization_to_covariance.py PolarizationToCovariance
```

Render a subset with the batch utility:

```bash
uv run --project . --extra native manimize-render --quality m --scene 03 --scene 04 --scene 11
```

## Rebuilding the included reference videos

The included videos were generated with the fallback renderer because the artifact-building sandbox did not contain the native Manim stack and could not download packages. The fallback reproduces the storyboard with Pillow, Matplotlib mathtext, and ffmpeg; it is **not** Manim and is kept separate from the official Manim source.

```bash
uv run --project . python tools/preview_renderer.py
```

## Kokoro-82M narration

Generate one WAV and one muxed MP4 for every numbered section. The default
`hold` fit preserves the animation speed and holds its final frame when speech
is longer; use `--fit stretch` when the whole animation should be retimed.

```bash
uv run --project . --extra narration manimize-narrate
```

Outputs are ignored under `audio/` and `timed_videos/`; the stitched narrated
series is `timed_videos/00_full_series_with_audio.mp4`. The source MP4s remain
unchanged, and the command prints the measured duration for every section.

The portfolio's Animated companion is built from that narration-timed series.
Its WebM and MP4 exports retain the generated voice track, and the native
Manim theme shares the site's ink, paper, muted, panel, and rule tones.

## Interpretation boundary

The visual series preserves the manuscripts' distinctions:

- information distance is **not** presented as a covariance estimate;
- the barycentric interaction field is **not** presented as a causal network;
- the portfolio certificate is **not** presented as an expected-return signal;
- transmission constants and slack are maintained sensitivity inputs, not quantities inferred from the animations;
- the three barycentric/coupling optimizations keep their different fixed and varying objects explicit.

## Customization

Edit `theme.py` for global visual changes. Every scene is otherwise self-contained. The scripts use synthetic low-dimensional clouds so that the geometry is visible; they do not reproduce the manuscripts' proprietary or generated empirical artifacts.
