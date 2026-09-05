"""Paper 5 target-specific natural-size figure measures."""

from pipeline.figures._common import TEXTWIDTH_IN

_MM_PER_INCH = 25.4

# Oxford University Press's `modern,small` class fixes `\textwidth` at 124mm.
# Express it as a fraction of the shared 5.906in authoring measure so Paper 5 can
# reuse the established `figsize`/`fit_to_width` contract without widening that
# shared API or invalidating unrelated figure stages.
JFEC_TEXTWIDTH_IN = 124.0 / _MM_PER_INCH
JFEC_WIDTH_FRACTION = JFEC_TEXTWIDTH_IN / TEXTWIDTH_IN
