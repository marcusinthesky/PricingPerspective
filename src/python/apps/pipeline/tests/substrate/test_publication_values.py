r"""Tests for the renderer-neutral publication-value model and its LaTeX emitter.

The model (:mod:`pipeline.io.values`) is one ordered ``list[Record]`` — mixing
``PublicationValue``, ``Comment``, and ``Blank`` records — that the LaTeX
``\\ppDeclareValue`` catalog is projected from. These tests pin the byte-level
output, the duplicate-key gate, the mixed-case key grammar, and the
missing/NaN policy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pipeline.io.values import (
    MISSING,
    Blank,
    Comment,
    PublicationValue,
    PublicationValueError,
    emit_latex,
    public_values,
)

if TYPE_CHECKING:
    from types import ModuleType

    from pipeline.io.values import Record

# --------------------------------------------------------------------------- #
# Scalar formatting
# --------------------------------------------------------------------------- #


def test_fixed_float_formatting() -> None:
    """Fixed-point values honor their declared precision."""
    pv = PublicationValue(key="sharpe-ratio", value=0.392967, precision=3)
    assert emit_latex([pv]) == "\\ppDeclareValue{sharpe-ratio}{0.393}\n"


def test_scientific_float_formatting() -> None:
    """Scientific values honor their declared precision."""
    pv = PublicationValue(key="p-value", value=0.0000031, precision=2, kind="e")
    assert emit_latex([pv]) == "\\ppDeclareValue{p-value}{3.10e-06}\n"


def test_general_float_formatting() -> None:
    """General-format values honor their declared precision."""
    # kind="g" (precision 6 == bare :g) drops trailing zeros: 1.0 -> "1".
    pvs: list[Record] = [
        PublicationValue(key="a", value=1.0, kind="g"),
        PublicationValue(key="b", value=1.5, kind="g"),
    ]
    assert emit_latex(pvs) == ("\\ppDeclareValue{a}{1}\n\\ppDeclareValue{b}{1.5}\n")


def test_integer_formatting() -> None:
    """Integer values render without float coercion."""
    pv = PublicationValue(key="n-obs", value=52)
    assert emit_latex([pv]) == "\\ppDeclareValue{n-obs}{52}\n"


def test_boolean_formatting() -> None:
    """Boolean values render as lowercase LaTeX-safe literals."""
    pvs: list[Record] = [
        PublicationValue(key="rejects-null", value=True),
        PublicationValue(key="rejects-alt", value=False),
    ]
    assert emit_latex(pvs) == (
        "\\ppDeclareValue{rejects-null}{true}\n\\ppDeclareValue{rejects-alt}{false}\n"
    )


def test_string_formatting_latex_escaped() -> None:
    """String bindings escape LaTeX control characters."""
    pv = PublicationValue(key="fit-date", value="2018-04-05")
    assert emit_latex([pv]) == "\\ppDeclareValue{fit-date}{2018-04-05}\n"


def test_tuple_formatting() -> None:
    """Tuple values render element-by-element in input order."""
    # LaTeX joins tuple elements with ", " (no parens).
    ints = PublicationValue(key="mc-n-assets", value=(10, 20, 53))
    floats = PublicationValue(key="mc-lipschitz-scales", value=(1.0, 1.5), kind="g")
    assert emit_latex([ints]) == "\\ppDeclareValue{mc-n-assets}{10, 20, 53}\n"
    assert emit_latex([floats]) == "\\ppDeclareValue{mc-lipschitz-scales}{1, 1.5}\n"


# --------------------------------------------------------------------------- #
# Escaping and rejection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("ledoit_wolf", r"ledoit\_wolf"),
        ("50% coverage", r"50\% coverage"),
        ("a & b", r"a \& b"),
        ("#tag", r"\#tag"),
        ("$5", r"\$5"),
        ("{x}", r"\{x\}"),
        ("a^b", r"a\textasciicircum{}b"),
        ("a~b", r"a\textasciitilde{}b"),
        ("a\\b", r"a\textbackslash{}b"),
    ],
)
def test_string_tex_special_escaping(raw: str, escaped: str) -> None:
    """Every supported TeX special receives its canonical escape."""
    pv = PublicationValue(key="label", value=raw)
    assert emit_latex([pv]) == f"\\ppDeclareValue{{label}}{{{escaped}}}\n"


def test_rejects_embedded_double_quote() -> None:
    """Quoted publication strings are rejected instead of under-escaped."""
    with pytest.raises(PublicationValueError):
        PublicationValue(key="bad", value='has "quote"')


def test_rejects_embedded_newline() -> None:
    """Publication strings cannot span generated binding lines."""
    with pytest.raises(PublicationValueError):
        PublicationValue(key="bad", value="line1\nline2")


def test_rejects_infinite_float() -> None:
    """Infinite floats cannot masquerade as missing values."""
    with pytest.raises(PublicationValueError):
        PublicationValue(key="bad", value=math.inf)


def test_rejects_unsupported_scalar_type() -> None:
    """Unsupported runtime scalar types fail at construction."""
    # A value outside bool/int/float/str/MISSING must be rejected at
    # construction, not silently accepted and mis-rendered (or crashing) at
    # emit time.
    with pytest.raises(PublicationValueError, match="unsupported value type"):
        PublicationValue(key="bad", value=object())  # pyrefly: ignore[bad-argument-type]  # deliberately invalid runtime type
    with pytest.raises(PublicationValueError, match="unsupported value type"):
        PublicationValue(key="bad-tuple", value=(1.0, object()))  # pyrefly: ignore[bad-argument-type]  # deliberately invalid tuple element


def test_rejects_underscore_prefixed_private_key() -> None:
    """Public bindings cannot use the private-key namespace."""
    with pytest.raises(PublicationValueError):
        PublicationValue(key="_u", value=1)


def test_rejects_non_ascii_or_illegal_key() -> None:
    """Public keys enforce the documented ASCII grammar."""
    for bad in ("Bad_Key", "a b", "a.b", "λ", "a/b"):
        with pytest.raises(PublicationValueError):
            PublicationValue(key=bad, value=1)


def test_preserves_mixed_case_public_keys() -> None:
    """Valid mixed-case keys retain their stable spelling."""
    # Stable mixed-case identifiers must NOT be lowercased or rejected.
    for key in ("h1-L-hat", "sample-T60-vol", "p4-L-hat", "robust-exact-T250-sharpe"):
        pv = PublicationValue(key=key, value=1.0, precision=1)
        assert emit_latex([pv]) == f"\\ppDeclareValue{{{key}}}{{1.0}}\n"


# --------------------------------------------------------------------------- #
# Missing / NaN policy
# --------------------------------------------------------------------------- #


def test_missing_and_nan_latex_ppmissing() -> None:
    """Missing and NaN values share the explicit missing-value macro."""
    recs: list[Record] = [
        PublicationValue(key="unmeasured", value=MISSING),
        PublicationValue(key="unmeasured-float", value=math.nan),
    ]
    assert emit_latex(recs) == (
        "\\ppDeclareValue{unmeasured}{\\ppmissing}\n"
        "\\ppDeclareValue{unmeasured-float}{\\ppmissing}\n"
    )


# --------------------------------------------------------------------------- #
# Comment / blank records
# --------------------------------------------------------------------------- #


def test_comment_and_blank_records() -> None:
    """Comment and blank records retain catalog structure."""
    recs: list[Record] = [
        Comment("AUTO-GENERATED by pipeline test — do not edit by hand."),
        Comment("Source: data/x.yaml"),
        Blank(),
        PublicationValue(key="n", value=1),
    ]
    assert emit_latex(recs) == (
        "% AUTO-GENERATED by pipeline test — do not edit by hand.\n"
        "% Source: data/x.yaml\n"
        "\n"
        "\\ppDeclareValue{n}{1}\n"
    )


def test_comment_preserves_leading_whitespace() -> None:
    """Comment content preserves intentional leading whitespace."""
    # A section-comment continuation with extra leading spaces must round-trip.
    rec = Comment("    calibration-fixed -> width is window-invariant ---")
    assert (
        emit_latex([rec])
        == "%     calibration-fixed -> width is window-invariant ---\n"
    )


def test_trailing_comment_not_rendered_in_latex() -> None:
    """Provenance-only trailing comments stay out of rendered bindings."""
    rec = PublicationValue(
        key="cell", value=MISSING, trailing_comment="NOT COMPUTED this pass"
    )
    # The provenance trailing comment is not rendered in the LaTeX catalog.
    assert emit_latex([rec]) == "\\ppDeclareValue{cell}{\\ppmissing}\n"


# --------------------------------------------------------------------------- #
# Duplicate-key rejection
# --------------------------------------------------------------------------- #


def test_rejects_duplicate_public_keys_latex() -> None:
    """Duplicate public keys fail before a catalog is emitted."""
    recs: list[Record] = [
        PublicationValue(key="k", value=1),
        PublicationValue(key="k", value=2),
    ]
    with pytest.raises(PublicationValueError, match="duplicate public keys"):
        emit_latex(recs)


def test_duplicate_comments_and_blanks_are_allowed() -> None:
    """Structural comments and blanks may repeat freely."""
    # Only PublicationValue keys must be unique; comments/blanks may repeat.
    recs: list[Record] = [
        Comment("x"),
        Blank(),
        Comment("x"),
        Blank(),
        PublicationValue(key="a", value=1),
    ]
    expected_comments = sum(isinstance(record, Comment) for record in recs)
    assert emit_latex(recs).count("% x") == expected_comments


# --------------------------------------------------------------------------- #
# Golden snapshot
# --------------------------------------------------------------------------- #


def _snapshot_records() -> list[Record]:
    return [
        Comment("AUTO-GENERATED by pipeline test-report — do not edit by hand."),
        Blank(),
        PublicationValue(key="n-obs", value=52),
        PublicationValue(key="sharpe-ratio", value=0.392967, precision=6),
        PublicationValue(key="p-value", value=3.1e-06, precision=3, kind="e"),
        PublicationValue(key="rejects-null", value=True),
        PublicationValue(key="sample-T60-vol", value=0.12, precision=2),
        PublicationValue(key="unmeasured", value=MISSING),
    ]


def test_golden_snapshot_latex() -> None:
    """Representative records match the governed LaTeX snapshot."""
    assert emit_latex(_snapshot_records()) == (
        "% AUTO-GENERATED by pipeline test-report — do not edit by hand.\n"
        "\n"
        "\\ppDeclareValue{n-obs}{52}\n"
        "\\ppDeclareValue{sharpe-ratio}{0.392967}\n"
        "\\ppDeclareValue{p-value}{3.100e-06}\n"
        "\\ppDeclareValue{rejects-null}{true}\n"
        "\\ppDeclareValue{sample-T60-vol}{0.12}\n"
        "\\ppDeclareValue{unmeasured}{\\ppmissing}\n"
    )


# --------------------------------------------------------------------------- #
# Ordering and well-formedness
# --------------------------------------------------------------------------- #


def test_deterministic_ordering_is_input_order() -> None:
    """Catalog output preserves caller-defined record ordering."""
    recs: list[Record] = [
        PublicationValue(key="zeta", value=1),
        PublicationValue(key="alpha", value=2),
        PublicationValue(key="mu", value=3),
    ]
    latex = emit_latex(recs)
    assert latex.index("zeta") < latex.index("alpha") < latex.index("mu")
    assert emit_latex(recs) == latex  # re-run is byte-identical


def test_public_values_returns_publication_values_in_input_order() -> None:
    """Filtering public values retains their relative input order."""
    recs: list[Record] = [
        Comment("header"),
        Blank(),
        PublicationValue(key="n-obs", value=52),
        PublicationValue(key="sharpe", value=0.392967, precision=3),
        PublicationValue(key="fit-date", value="2018-04-05"),
        PublicationValue(key="grid", value=(3, 5, 10)),
        PublicationValue(key="unmeasured", value=MISSING),
    ]
    keys = [pv.key for pv in public_values(recs)]
    assert keys == ["n-obs", "sharpe", "fit-date", "grid", "unmeasured"]
    latex = _latex_bindings(emit_latex(recs))
    assert set(latex) == set(keys)
    assert latex["n-obs"] == "52"
    assert latex["sharpe"] == "0.393"
    assert latex["fit-date"] == "2018-04-05"
    assert latex["grid"] == "3, 5, 10"
    assert latex["unmeasured"] == r"\ppmissing"


def test_latex_emitter_never_emits_partial_or_empty_binding() -> None:
    """Rendered catalogs contain only complete nonempty bindings."""
    latex = emit_latex([PublicationValue(key="only-key", value=1)])
    for line in latex.splitlines():
        if not line.startswith("\\ppDeclareValue{"):
            continue
        inner = line[len("\\ppDeclareValue{") : -1]
        key, _, body = inner.partition("}{")
        assert key != ""
        assert body != ""
    assert "\\ppDeclareValue{}" not in latex
    assert "\\ppDeclareValue{}{}" not in latex


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _latex_bindings(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("\\ppDeclareValue{"):
            inner = line[len("\\ppDeclareValue{") : -1]
            k, v = inner.split("}{", 1)
            out[k] = v
    return out


def _reproduction_checker() -> ModuleType:
    name = "publication_reproduction_check"
    if name not in sys.modules:
        path = (
            Path(__file__).resolve().parents[4]
            / "scripts/replication_release/script.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_reproduction_comments_preserve_escaped_percent_and_later_numbers() -> None:
    """Escaped percentages remain visible and genuine comments end their line."""
    check = _reproduction_checker()
    text = r"12\% and 34 % hidden 99" + "\n" + r"56 \\% hidden 78"
    assert check.NUMBER.findall(check.uncomment(text)) == ["12", "34", "56"]


@pytest.mark.parametrize(
    ("text", "number", "expected"),
    [
        (r"\ppnum{fit-d2560}", "0.12345678901234567", "0.123"),
        (r"\ppnum{fit-d2560}", "0.12345678901234568", "0.123"),
        (r"\ppnum{fit-d2560}", "1.2345", "1.235"),
        (r"\ppnum{fit-d2560}", "-1.2345", "-1.235"),
        (r"\ppnum[5]{fit-d2560}", "0.12345678901234567", "0.12346"),
        (r"\ppnume[2]{fit-d2560}", "0.00000009995", "1E-7"),
        (r"\ppnume[2]{fit-d2560}", "0.00004768643914868921", "0.0000477"),
        (r"\ppvalue{fit-d2560}", "0.12345678901234567", "0.12345678901234567"),
        (
            r"\num[round-mode=places,round-precision=2]{\fpeval{100*\ppvalue{fit-d2560}}}",
            "0.12995",
            "13",
        ),
    ],
)
def test_reproduction_uses_actual_decimal_and_scientific_callsite_precision(
    text: str, number: str, expected: str
) -> None:
    """Formatting follows each call rather than the stored numeric precision."""
    check = _reproduction_checker()
    sites, literals = check.display_sites(text, "paper.tex")
    assert len(sites) == 1
    assert literals == []
    assert check.rendered_value(sites[0], {"fit-d2560": number}) == expected


def test_reproduction_ignores_unprinted_ulps_but_checks_printed_values() -> None:
    """Unprinted ULP differences pass while changed printed results fail."""
    check = _reproduction_checker()
    site = check.display_sites(r"\ppnum{fit}", "paper.tex")[0][0]
    baseline = check.rendered_value(site, {"fit": "0.12345678901234567"})
    assert check.rendered_value(site, {"fit": "0.12345678901234568"}) == baseline
    assert check.rendered_value(site, {"fit": "0.12450000000000000"}) != baseline


def test_reproduction_counts_and_sample_membership_remain_exact() -> None:
    """Count, date and sample membership checks preserve exact identity."""
    check = _reproduction_checker()
    expected = {
        "literal_numbers": ["2"],
        "integer_bindings": {"count": "2"},
        "date_bindings": {"date": "2018-01-01"},
        "cohort": "AAA BBB 2",
    }
    same = r"\ppDeclareValue{count}{2}\ppDeclareValue{date}{2018-01-01}" + "AAA BBB 2"
    assert check.fragment_differences("table.tex", same, expected) == []
    changed = same.replace("{count}{2}", "{count}{3}").replace("BBB", "CCC")
    failures = check.fragment_differences("table.tex", changed, expected)
    assert any("exact count" in failure for failure in failures)
    assert any("cohort membership" in failure for failure in failures)


def _scientific_curve(endpoint: str) -> str:
    return "\n".join(
        [
            r"\begin{pgfpicture}",
            r"\pgfpathrectangle{\pgfqpoint{0.000000in}{0.000000in}}{\pgfqpoint{4.000000in}{3.000000in}}",
            r"\pgfusepath{use as bounding box, clip}",
            r"\begin{pgfscope}",
            r"\pgfpathrectangle{\pgfqpoint{0.500000in}{0.500000in}}{\pgfqpoint{3.000000in}{2.000000in}}",
            r"\pgfusepath{clip}",
            r"\pgfpathmoveto{\pgfqpoint{1.000000in}{1.000000in}}",
            r"\pgfpathcurveto{\pgfqpoint{1.500000in}{1.500000in}}{\pgfqpoint{2.000000in}{1.250000in}}{\pgfqpoint{"
            + endpoint
            + r"in}{2.000000in}}",
            r"\pgfusepath{stroke}",
            r"\end{pgfscope}",
            r"\end{pgfpicture}",
        ]
    )


def test_reproduction_checks_scientific_curve_not_only_axis_labels() -> None:
    """A changed curve fails even when every plot label is unchanged."""
    check = _reproduction_checker()
    before = check.pgf_signature(_scientific_curve("2.500000"))
    after = check.pgf_signature(_scientific_curve("2.510000"))
    assert check.plot_differences(before, before) == []
    assert any(
        "curve/point coordinates changed" in difference
        for difference in check.plot_differences(before, after)
    )


def test_reproduction_rejects_unknown_scientific_path_command() -> None:
    """Unsupported geometry must fail instead of silently escaping comparison."""
    check = _reproduction_checker()
    text = _scientific_curve("2.500000").replace(r"\pgfpathcurveto", r"\pgfpatharcto")
    with pytest.raises(ValueError, match="unsupported PGF path command"):
        check.pgf_signature(text)


def test_reproduction_raster_comparison_ignores_metadata_but_checks_pixels() -> None:
    """PNG metadata is irrelevant but every scientific pixel remains checked."""
    check = _reproduction_checker()
    image = Image.new("RGB", (2, 2), "white")
    first = BytesIO()
    image.save(first, format="PNG")
    metadata = PngInfo()
    metadata.add_text("Software", "different harmless encoder label")
    second = BytesIO()
    image.save(second, format="PNG", pnginfo=metadata)
    assert check.raster_signature(first.getvalue()) == check.raster_signature(
        second.getvalue()
    )
    image.putpixel((0, 0), (0, 0, 0))
    third = BytesIO()
    image.save(third, format="PNG")
    assert check.raster_signature(first.getvalue()) != check.raster_signature(
        third.getvalue()
    )


def test_reproduction_operator_accepts_only_unprinted_precision_changes(
    tmp_path: Path,
) -> None:
    """The default complete comparison tolerates ULPs and rejects visible changes."""
    check = _reproduction_checker()
    fragment = "src/latex/projects/paper/src/generated/numbers.tex"
    consumer = "src/latex/projects/paper/src/chapters/results.tex"
    original = r"\ppDeclareValue{fit}{0.12345678901234567}\ppDeclareValue{n}{2}"
    fixed = json.dumps({fragment: original}).encode()
    fixed_path = tmp_path / "data/reference/published_values.json"
    fixed_path.parent.mkdir(parents=True)
    fixed_path.write_bytes(fixed)
    current_path = tmp_path / fragment
    current_path.parent.mkdir(parents=True)
    current_path.write_text(original.replace("34567", "34568"))
    consumer_path = tmp_path / consumer
    consumer_path.parent.mkdir(parents=True)
    consumer_path.write_text(r"\ppnum{fit}")
    site = check.display_sites(consumer_path.read_text(), consumer)[0][0]
    baseline = {
        "values_sha256": hashlib.sha256(fixed).hexdigest(),
        "fragments": {
            fragment: {
                "literal_numbers": list[str](),
                "integer_bindings": {"n": "2"},
                "date_bindings": dict[str, str](),
            }
        },
        "displays": {consumer: [dict(site.record(), expected="0.123")]},
        "figures": dict[str, object](),
    }
    failures, diagnostics, counts = check.check_results(
        tmp_path, baseline, raw_diagnostic=True
    )
    assert failures == []
    assert len(diagnostics) == 1
    assert counts["displays"] == 1
    current_path.write_text(original.replace("0.12345678901234567", "0.1245"))
    failures, _, _ = check.check_results(tmp_path, baseline)
    assert any("display #1" in failure and "0.125" in failure for failure in failures)
