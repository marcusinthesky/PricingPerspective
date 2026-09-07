# Package manifest

This manifest records the generated source, reference-render, and documentation artifacts.

## Reference videos

| File | Scene | Duration | Resolution | FPS | Size |
|---|---|---:|---:|---:|---:|
| `videos/00_full_series.mp4` | Complete 13-scene series | 97.5s | 1280×720 | 20/1 | 1303.7 KiB |
| `videos/01_firm_as_distribution.mp4` | A firm as a probability law | 7.5s | 1280×720 | 20/1 | 83.7 KiB |
| `videos/02_couplings_and_wasserstein.mp4` | Couplings and quadratic transport | 7.5s | 1280×720 | 20/1 | 161.3 KiB |
| `videos/03_polarization_to_covariance.mp4` | Polarization: distance becomes covariance | 7.5s | 1280×720 | 20/1 | 101.2 KiB |
| `videos/04_covariance_envelope.mp4` | A sharp covariance envelope | 7.5s | 1280×720 | 20/1 | 64.1 KiB |
| `videos/05_transmission_and_slack.mp4` | Transmission, distortion and slack | 7.5s | 1280×720 | 20/1 | 101.6 KiB |
| `videos/06_target_anchored_reconstruction.mp4` | Target-anchored barycentric reconstruction | 7.5s | 1280×720 | 20/1 | 83.1 KiB |
| `videos/07_symmetric_distance_directed_field.mp4` | A symmetric metric can yield a directed field | 7.5s | 1280×720 | 20/1 | 91.7 KiB |
| `videos/08_quadratic_adjustment_spatial_closure.mp4` | Quadratic adjustment implies spatial closure | 7.5s | 1280×720 | 20/1 | 85.6 KiB |
| `videos/09_pairwise_not_jointly_coherent.mp4` | Pairwise plans need not form one joint law | 7.5s | 1280×720 | 20/1 | 111.5 KiB |
| `videos/10_multifirm_dispersion_barycentre.mp4` | Multi-firm transport dispersion | 7.5s | 1280×720 | 20/1 | 81.4 KiB |
| `videos/11_portfolio_variance_certificate.mp4` | Portfolio variance as alignment minus dispersion | 7.5s | 1280×720 | 20/1 | 123.9 KiB |
| `videos/12_certificate_convexity.mp4` | Geometry can certify convex optimization | 7.5s | 1280×720 | 20/1 | 98.8 KiB |
| `videos/13_three_optimizations.mp4` | Three distinct optimization layers | 7.5s | 1280×720 | 20/1 | 123.2 KiB |

## Manim source files

| File | Scene class | Size |
|---|---|---:|
| `scenes/01_firm_as_distribution.py` | `FirmAsDistribution` | 2.7 KiB |
| `scenes/02_couplings_and_wasserstein.py` | `CouplingsAndWasserstein` | 2.4 KiB |
| `scenes/03_polarization_to_covariance.py` | `PolarizationToCovariance` | 3.2 KiB |
| `scenes/04_covariance_envelope.py` | `CovarianceEnvelope` | 3.2 KiB |
| `scenes/05_transmission_and_slack.py` | `TransmissionAndSlack` | 3.4 KiB |
| `scenes/06_target_anchored_reconstruction.py` | `TargetAnchoredReconstruction` | 3.7 KiB |
| `scenes/07_symmetric_distance_directed_field.py` | `SymmetricDistanceDirectedField` | 3.5 KiB |
| `scenes/08_quadratic_adjustment_spatial_closure.py` | `QuadraticAdjustmentSpatialClosure` | 2.7 KiB |
| `scenes/09_pairwise_not_jointly_coherent.py` | `PairwiseNotJointlyCoherent` | 3.3 KiB |
| `scenes/10_multifirm_dispersion_barycentre.py` | `MultifirmDispersionBarycentre` | 3.5 KiB |
| `scenes/11_portfolio_variance_certificate.py` | `PortfolioVarianceCertificate` | 3.3 KiB |
| `scenes/12_certificate_convexity.py` | `CertificateConvexity` | 2.9 KiB |
| `scenes/13_three_optimizations.py` | `ThreeOptimizations` | 5.0 KiB |

## Supporting files

| File | Purpose | Size |
|---|---|---:|
| `README.md` | Project overview and rendering instructions | 6.2 KiB |
| `STORYBOARD.md` | Per-scene visual sequence | 6.6 KiB |
| `NARRATION.md` | Voice-over copy consumed by the Kokoro-82M pipeline | 6.0 KiB |
| `AUDIENCE_GUIDE.md` | Presentation paths by audience | 1.8 KiB |
| `SOURCE_NOTES.md` | Manuscript equation and theorem anchors | 4.5 KiB |
| `BUILD_INFO.md` | Native-source versus reference-render disclosure | 1.4 KiB |
| `gallery.html` | Offline browser gallery | 6.6 KiB |
| `theme.py` | Shared Manim styling helpers | 2.8 KiB |
| `src/manimize/render_all.py` | Native Manim batch renderer | 6.2 KiB |
| `src/manimize/narration.py` | Kokoro synthesis and narration timing | 9.9 KiB |
| `tests/test_narration.py` | Narration parser and stem tests | 1.1 KiB |
| `justfile` | App quality and rendering commands | 1.0 KiB |
| `pyproject.toml` | uv project and quality-gate configuration | 2.3 KiB |
| `manim.cfg` | Manim project configuration | 0.1 KiB |
| `chapters.ffmeta` | Combined-video chapter metadata | 1.4 KiB |
| `LICENSE` | Code license and manuscript-rights boundary | 1.2 KiB |
| `palette.py` | Portfolio-neutral palette shared by renderers | 0.4 KiB |
| `thumbnails/contact_sheet.png` | All-scene visual QA/contact sheet | 471.1 KiB |
| `tools/preview_renderer.py` | Deterministic vector reference renderer | 45.5 KiB |
| `videos/README.md` | Video-folder disclosure | 0.4 KiB |
| `scenes/README.md` | Direct scene-render command | 0.5 KiB |

## Counts

- Manim scene scripts: **13**
- Standalone reference videos: **13**
- Combined reference videos: **1**
- PNG thumbnails, including contact sheet: **14**

Integrity hashes are recorded in `SHA256SUMS.txt`.
