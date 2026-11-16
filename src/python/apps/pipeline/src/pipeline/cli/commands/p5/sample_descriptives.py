"""Paper 5 per-sector sample descriptives table (shared compute, P5 render)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()

_P5_DESCRIPTIVES_CAPTION = (
    r"Per-sector composition of the news frame and its balanced-cloud sample: "
    r"firm count, mean article count per firm, mean daily "
    r"return volatility, and mean within- versus cross-sector rooted $W_2$ "
    r"distance under the frozen Qwen3-Embedding-8B geometry. Firm counts, "
    r"volatility, and distance summaries use the same estimation sample. "
    r"The Articles column reports the constant balanced-cloud "
    r"size after per-firm truncation, not raw frame coverage, and therefore "
    r"carries no information about a firm's true news volume. Volatility is "
    r"daily and unannualized. A dash marks a statistic "
    r"undefined for that sector, such as within-sector distance for a "
    r"singleton."
)


@commands.command("render-paper5-sample-descriptives")
def cli_render_paper5_sample_descriptives(
    summary_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Sample-descriptives results.json under Paper 5's encoder.",
        ),
    ],
    output_file: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated table fragment.")
    ],
) -> None:
    """Render Paper 5's per-sector sample-descriptives table.

    Shares Paper 1's tabulation and renderer; only the encoder identity, the
    caption and the float label differ, so the two papers cannot collide on a
    label while remaining independently self-contained.
    """
    from pipeline.figures.paper1.sample_descriptives import (
        render_sample_descriptives_table,
    )

    # Inputs pass positionally so the shared renderer's parameter name is free
    # to follow its file format. The generator string is frozen with the
    # published fragment's header comment.
    render_sample_descriptives_table(
        summary_file,
        output_file,
        caption=_P5_DESCRIPTIVES_CAPTION,
        label="tab:p5-sample-descriptives",
        generator="pipeline render-paper5-descriptives",
    )
