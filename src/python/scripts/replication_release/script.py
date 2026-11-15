#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "~=3.13.0"
# dependencies = ["Pillow==12.3.0"]
# ///
"""Compare displayed manuscript results and figures at publication precision."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, replace
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

from PIL import Image

_MAX_EXPONENT = 20
_RGB_COMPONENTS = 3
_POINT_COMPONENTS = 2
_MAX_FIGURE_FAILURES = 8

type Point = tuple[float, float]
type Frame = tuple[float, float, float, float]
type Commands = list[tuple[str, list[Point]]]

NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
SCALAR = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\Z")
INTEGER = re.compile(r"[-+]?\d+\Z")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def uncomment(text: str) -> str:
    """Remove TeX comments while retaining escaped percent signs and line breaks."""
    result = []
    for original in text.splitlines():
        line = original
        for index, character in enumerate(original):
            if character != "%":
                continue
            backslashes = 0
            previous = index - 1
            while previous >= 0 and line[previous] == "\\":
                backslashes += 1
                previous -= 1
            if backslashes % 2 == 0:
                line = line[:index]
                break
        result.append(line)
    return "\n".join(result)


def group(text: str, start: int, opening: str = "{") -> tuple[str, int]:
    """Read one balanced TeX argument, including nested value references."""
    closing = "}" if opening == "{" else "]"
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != opening:
        message = f"expected {opening!r} near {text[start : start + 50]!r}"
        raise ValueError(message)
    depth = 1
    index = start + 1
    while index < len(text):
        if text[index] == "\\" and text[index + 1 : index + 2] in "{}[]%":
            index += 2
            continue
        if text[index] == opening:
            depth += 1
        if text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
        index += 1
    message = f"unclosed {opening!r} argument"
    raise ValueError(message)


def declarations(text: str) -> tuple[dict[str, str], str]:
    """Separate value declarations from the printed body of a generated fragment."""
    text = uncomment(text)
    values = {}
    spans = []
    for match in re.finditer(r"\\ppDeclareValue\b", text):
        key, end = group(text, match.end())
        value, end = group(text, end)
        if key in values and values[key] != value:
            message = f"conflicting generated binding {key!r}"
            raise ValueError(message)
        values[key] = value
        spans.append((match.start(), end))
    for start, end in reversed(spans):
        text = text[:start] + " " * (end - start) + text[end:]
    return values, text


@dataclass(frozen=True)
class Display:
    """One actual manuscript display, independent of the catalog's stored precision."""

    source: str
    ordinal: int
    expression: str
    mode: str = "off"
    precision: int = 3
    scientific: bool = False

    def record(self) -> dict[str, str | int | bool]:
        """Return a portable baseline record without machine or repository metadata."""
        return asdict(self)


def display_sites(text: str, source: str) -> tuple[list[Display], list[str]]:
    """Find formatted calls and bare values; return remaining literal table numbers."""
    _, text = declarations(text)
    occupied = []
    sites = []
    pattern = re.compile(r"\\(ppnume|ppnum|num|fpeval)\b")
    for match in pattern.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        options = ""
        if text[end : end + 1] == "[":
            options, end = group(text, end, "[")
        expression, end = group(text, end)
        command = match[1]
        if command.startswith("ppnum"):
            expression = r"\ppvalue{" + expression + "}"
            mode = "places"
            precision = int(options or "3")
            scientific = command == "ppnume"
        else:
            settings = dict(
                part.strip().split("=", 1) for part in options.split(",") if "=" in part
            )
            mode = settings.get("round-mode", "off")
            precision = int(settings.get("round-precision", "3"))
            scientific = settings.get("scientific-notation", "false") == "true"
            if command == "fpeval":
                expression = r"\fpeval{" + expression + "}"
        sites.append(
            Display(source, match.start(), expression, mode, precision, scientific)
        )
        occupied.append((match.start(), end))
    for match in re.finditer(r"\\ppvalue\b", text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        expression, end = group(text, match.end())
        sites.append(Display(source, match.start(), r"\ppvalue{" + expression + "}"))
        occupied.append((match.start(), end))
    sites.sort(key=lambda site: site.ordinal)
    sites = [
        Display(
            source, index, site.expression, site.mode, site.precision, site.scientific
        )
        for index, site in enumerate(sites)
    ]
    for start, end in sorted(occupied, reverse=True):
        text = text[:start] + " " * (end - start) + text[end:]
    return sites, NUMBER.findall(text)


def _arithmetic(expression: str) -> Decimal:  # noqa: C901 - explicit safe AST grammar
    """Evaluate only decimal arithmetic, never arbitrary Python or TeX commands."""
    expression = expression.replace("^", "**").strip()
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> Decimal:  # noqa: C901, PLR0911 - one case per allowed operator
        match node:
            case ast.Constant(value=int() | float()):
                return Decimal(ast.get_source_segment(expression, node))
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return -evaluate(operand)
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return evaluate(operand)
            case ast.BinOp(left=left, op=operator, right=right):
                first, second = evaluate(left), evaluate(right)
                match operator:
                    case ast.Add():
                        return first + second
                    case ast.Sub():
                        return first - second
                    case ast.Mult():
                        return first * second
                    case ast.Div():
                        return first / second
                    case ast.Pow() if (
                        second == second.to_integral_value()
                        and abs(second) <= _MAX_EXPONENT
                    ):
                        return first ** int(second)
            case ast.Call(func=ast.Name(id="sqrt"), args=[argument], keywords=[]):
                return evaluate(argument).sqrt()
            case ast.Call(func=ast.Name(id="abs"), args=[argument], keywords=[]):
                return abs(evaluate(argument))
        message = f"unsupported numeric expression: {expression!r}"
        raise ValueError(message)

    with localcontext() as context:
        context.prec = 80
        return evaluate(tree.body)


def evaluate_expression(expression: str, values: dict[str, str]) -> str:
    """Substitute catalog values and evaluate explicit fpeval arithmetic."""

    def replace(match: re.Match[str]) -> str:
        key = match[1]
        if key not in values:
            message = f"missing generated binding {key!r}"
            raise ValueError(message)
        return values[key]

    expression = re.sub(r"\\ppvalue\{([^{}]+)\}", replace, expression).strip()
    if expression.startswith(r"\fpeval{"):
        body, end = group(expression, len(r"\fpeval"))
        if end != len(expression):
            message = f"trailing content after fpeval: {expression!r}"
            raise ValueError(message)
        return str(_arithmetic(body))
    return expression


def rendered_value(display: Display, values: dict[str, str]) -> str:
    """Apply siunitx decimal places/significant figures and its default half-up rule."""
    value = evaluate_expression(display.expression, values)
    if not SCALAR.fullmatch(value):
        return value
    number = Decimal(value)
    if display.mode == "off" or not number:
        return str(number.normalize())
    if display.mode == "figures":
        power = number.adjusted() - display.precision + 1
    elif display.mode == "places":
        power = (
            number.adjusted() - display.precision
            if display.scientific
            else -display.precision
        )
    else:
        message = f"unsupported rounding mode {display.mode!r}"
        raise ValueError(message)
    with localcontext() as context:
        context.prec = 80
        rounded = number.quantize(Decimal(1).scaleb(power), rounding=ROUND_HALF_UP)
    return str(rounded.normalize()) if rounded else "0"


POINT = re.compile(r"\\pgfqpoint\{([-+0-9.eE]+)in\}\{([-+0-9.eE]+)in\}")
SHIFT = re.compile(r"\\pgfsys@transformshift\{([-+0-9.eE]+)in\}\{([-+0-9.eE]+)in\}")
COLOR = re.compile(r"\\definecolor\{(currentfill|currentstroke)\}\{rgb\}\{([^}]+)\}")
TEXT_PREFIX = r"\def%{\%}"
PROVENANCE = re.compile(
    r"provenance|fingerprint|sha256|code.identity|source.digest", re.IGNORECASE
)


@dataclass
class State:
    """Graphics state inherited across nested PGF scopes."""

    clip: tuple[float, float, float, float] | None = None
    fill: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stroke: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shift: tuple[float, float] | None = None


def raster_signature(data: bytes) -> dict:
    """Hash canonical RGBA pixels, ignoring PNG encoding and ancillary metadata."""
    with Image.open(io.BytesIO(data)) as image:
        normalized = image.convert("RGBA")
        return {
            "size": list(normalized.size),
            "mode": "RGBA",
            "pixels_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
        }


def pgf_signature(text: str, *, schematic: bool = False) -> dict:  # noqa: C901, PLR0912, PLR0915 - explicit serializer grammar
    """Read clipped scientific paths and markers in panel coordinates.

    Matplotlib serializes coordinates in inches to six decimal places. Panel
    normalization removes text-driven translations/resizing. Numeric labels and
    fill colors retain absolute scales and vector heatmap values, respectively.
    Schematic paths use the figure canvas as their dimensionless frame.
    """
    states = [State()]
    pending = []
    rectangle = None
    definition = False
    canvas = None
    panels = []
    primitives = []
    numeric_labels = []

    def panel_index(rect: Frame) -> int:
        if rect not in panels:
            panels.append(rect)
        return panels.index(rect)

    def normalized(commands: Commands, rect: Frame) -> list[list[str | float]]:
        x, y, width, height = rect
        output = []
        for command, points in commands:
            output.append(
                [
                    command,
                    *[
                        component
                        for px, py in points
                        for component in ((px - x) / width, (py - y) / height)
                    ],
                ]
            )
        return output

    def emit(kind: str, commands: Commands, mode: str = "") -> None:
        state = states[-1]
        if state.shift is not None:
            dx, dy = state.shift
            commands = [
                (name, [(px + dx, py + dy) for px, py in points])
                for name, points in commands
            ]
        frame = state.clip or (canvas if schematic else None)
        if frame is None:
            return
        if frame[2] <= 0 or frame[3] <= 0:
            message = "non-positive PGF panel extent"
            raise ValueError(message)
        item = {
            "kind": kind,
            "panel": panel_index(frame),
            "commands": normalized(commands, frame),
        }
        if "fill" in mode:
            item["fill"] = list(state.fill)
        if "stroke" in mode:
            item["stroke"] = list(state.stroke)
        primitives.append(item)

    for line in text.splitlines():
        if line.startswith("%%"):
            continue
        for name in re.findall(r"\\(pgfpath[A-Za-z@]+)", line):
            if name not in {
                "pgfpathrectangle",
                "pgfpathmoveto",
                "pgfpathlineto",
                "pgfpathcurveto",
                "pgfpathquadraticcurveto",
                "pgfpathclose",
            }:
                message = f"unsupported PGF path command: {name}"
                raise ValueError(message)
        for name in re.findall(r"\\(pgf(?:sys@)?transform[A-Za-z@]*)", line):
            if name != "pgfsys@transformshift":
                message = f"unsupported PGF transform: {name}"
                raise ValueError(message)
        if line.startswith(r"\begin{pgfscope}"):
            states.append(replace(states[-1]))
            continue
        if line.startswith(r"\end{pgfscope}"):
            if len(states) == 1:
                message = "unbalanced PGF scopes"
                raise ValueError(message)
            states.pop()
            continue
        if line.startswith(r"\pgfsys@defobject"):
            definition = True
            continue
        if definition:
            if line.startswith("}%"):
                definition = False
            continue
        state = states[-1]
        color = COLOR.match(line)
        if color:
            values = tuple(map(float, color[2].split(",")))
            if len(values) != _RGB_COMPONENTS or any(
                not math.isfinite(v) or not 0 <= v <= 1 for v in values
            ):
                message = "invalid RGB scientific color"
                raise ValueError(message)
            setattr(state, "fill" if color[1] == "currentfill" else "stroke", values)
            continue
        if line.startswith(
            (r"\definecolor{currentfill}", r"\definecolor{currentstroke}")
        ):
            message = "unsupported scientific color encoding"
            raise ValueError(message)
        shift = SHIFT.match(line)
        if shift:
            dx, dy = state.shift or (0.0, 0.0)
            state.shift = (dx + float(shift[1]), dy + float(shift[2]))
            continue
        if line.startswith(r"\pgfsys@transformshift"):
            message = "unsupported PGF translation coordinates"
            raise ValueError(message)
        if line.startswith(r"\pgfsys@useobject{currentpattern}"):
            continue
        if line.startswith(r"\pgfsys@useobject"):
            if not line.startswith(r"\pgfsys@useobject{currentmarker}"):
                message = "unsupported PGF graphical object"
                raise ValueError(message)
            if state.shift is None:
                message = "PGF marker without a position"
                raise ValueError(message)
            emit("marker", [("point", [(0.0, 0.0)])], "fill,stroke")
            continue
        if line.startswith(r"\pgftext") and TEXT_PREFIX in line:
            payload = line.split(TEXT_PREFIX, 1)[1]
            if not PROVENANCE.search(payload):
                numbers = NUMBER.findall(payload)
                if numbers:
                    numeric_labels.append(
                        [str(Decimal(value).normalize()) for value in numbers]
                    )
            continue
        if line.startswith(r"\pgfpathrectangle"):
            points = [tuple(map(float, p)) for p in POINT.findall(line)]
            if r"\pgfpointorigin" in line:
                x, y = 0.0, 0.0
                width, height = points[0]
            elif len(points) == _POINT_COMPONENTS:
                (x, y), (width, height) = points
            else:
                message = "unsupported PGF rectangle"
                raise ValueError(message)
            rectangle = (x, y, width, height)
            if canvas is None:
                canvas = rectangle
            pending = [("rectangle", [(x, y), (x + width, y + height)])]
            continue
        command = re.match(r"\\pgfpath(moveto|lineto|curveto|quadraticcurveto)\b", line)
        if command:
            points = [tuple(map(float, p)) for p in POINT.findall(line)]
            expected = {"moveto": 1, "lineto": 1, "curveto": 3, "quadraticcurveto": 2}[
                command[1]
            ]
            if len(points) != expected:
                message = "unsupported or incomplete PGF path coordinates"
                raise ValueError(message)
            if any(not math.isfinite(v) for point in points for v in point):
                message = "nonfinite PGF coordinate"
                raise ValueError(message)
            pending.append((command[1], points))
            continue
        if line.startswith(r"\pgfpathclose"):
            pending.append(("close", []))
            continue
        use = re.match(r"\\pgfusepath\{([^}]+)\}", line)
        if use:
            mode = use[1]
            if "clip" in mode:
                if "bounding box" not in mode:
                    if rectangle is None:
                        if state.clip is None or not pending:
                            message = "unsupported non-rectangular clip"
                            raise ValueError(message)
                        emit("clip-shape", pending)
                    else:
                        dx, dy = state.shift or (0.0, 0.0)
                        x, y, w, h = rectangle
                        state.clip = (x + dx, y + dy, w, h)
            elif pending:
                emit("path", pending, mode)
            pending = []
            rectangle = None
    if len(states) != 1 or definition:
        message = "incomplete PGF scope or marker definition"
        raise ValueError(message)
    if not primitives:
        message = "no scientific PGF paths or markers found"
        raise ValueError(message)
    return {
        "schematic": schematic,
        "panels": [[p[2], p[3]] for p in panels],
        "primitives": primitives,
        "numeric_labels": numeric_labels,
    }


def plot_differences(before: dict, after: dict) -> list[str]:  # noqa: C901, PLR0912 - compare every serialized primitive
    """Compare scientific geometry at the PGF renderer's one-microinch resolution."""
    failures = []
    if before["numeric_labels"] != after["numeric_labels"]:
        failures.append("numeric plot labels/scales changed")
    if len(before["panels"]) != len(after["panels"]):
        failures.append("scientific panel count changed")
    expected = before["primitives"]
    actual = after["primitives"]
    if len(expected) != len(actual):
        failures.append(
            f"scientific path/marker count changed: {len(expected)} -> {len(actual)}"
        )
        return failures
    for index, (old, new) in enumerate(zip(expected, actual, strict=True)):
        if old["kind"] != new["kind"] or old["panel"] != new["panel"]:
            failures.append(f"scientific primitive {index} changed kind/panel")
            continue
        failures.extend(
            f"scientific primitive {index} {color} values changed"
            for color in ("fill", "stroke")
            if old.get(color) != new.get(color)
        )
        if len(old["commands"]) != len(new["commands"]):
            failures.append(
                f"scientific primitive {index} changed its curve/path length"
            )
            continue
        panel = old["panel"]
        if panel >= len(after["panels"]):
            continue
        widths = before["panels"][panel]
        new_widths = after["panels"][panel]
        changed = False
        for first, second in zip(old["commands"], new["commands"], strict=True):
            if first[0] != second[0] or len(first) != len(second):
                changed = True
                break
            for coordinate, (left, right) in enumerate(
                zip(first[1:], second[1:], strict=True)
            ):
                # Each serialized point and clip bound is rounded to 1e-6 inch.
                # Both normalizations contribute to the error envelope.
                tolerance = (
                    (2.0 + max(abs(left), abs(right)))
                    * 1e-6
                    / min(widths[coordinate % 2], new_widths[coordinate % 2])
                )
                if abs(left - right) > tolerance:
                    changed = True
                    break
            if changed:
                break
        if changed:
            failures.append(
                f"scientific primitive {index} curve/point coordinates changed"
            )
        if len(failures) >= _MAX_FIGURE_FAILURES:
            break
    return failures


def _paper(relative: str) -> str:
    return relative.split("/src/", 1)[0]


def _catalogs(fragments: dict[str, str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative, text in fragments.items():
        values, _ = declarations(text)
        catalog = result.setdefault(_paper(relative), {})
        for key, value in values.items():
            if key in catalog and catalog[key] != value:
                message = f"{relative}: conflicting generated binding {key!r}"
                raise ValueError(message)
            catalog[key] = value
    return result


def _display_record(record: dict[str, object]) -> Display:
    return Display(**{key: value for key, value in record.items() if key != "expected"})


def fragment_differences(
    relative: str, text: str, expected: dict[str, object]
) -> list[str]:
    """Check printed literals, exact counts/dates and complete sample membership."""
    values, body = declarations(text)
    _, literal_numbers = display_sites(text, relative)
    failures = []
    if literal_numbers != expected["literal_numbers"]:
        before = expected["literal_numbers"]
        for index, (first, second) in enumerate(
            zip(before, literal_numbers, strict=False)
        ):
            if first != second:
                failures.append(
                    f"{relative}: table literal #{index + 1}: {first!r} -> {second!r}"
                )
                break
        else:
            failures.append(
                f"{relative}: printed table literal count changed: "
                f"{len(before)} -> {len(literal_numbers)}"
            )
    for key, wanted in expected["integer_bindings"].items():
        actual = values.get(key)
        if (
            actual is None
            or not SCALAR.fullmatch(actual)
            or Decimal(actual) != Decimal(wanted)
        ):
            failures.append(
                f"{relative}: exact count {key!r}: {wanted!r} -> {actual!r}"
            )
    for key, wanted in expected["date_bindings"].items():
        if values.get(key) != wanted:
            failures.append(
                f"{relative}: cohort date {key!r}: {wanted!r} -> {values.get(key)!r}"
            )
    if "cohort" in expected and re.sub(r"\s+", " ", body).strip() != expected["cohort"]:
        failures.append(f"{relative}: sample cohort membership or ordering changed")
    return failures


def check_results(  # noqa: C901, PLR0912 - report all independent failures
    root: Path, baseline: dict[str, object], *, raw_diagnostic: bool = False
) -> tuple[list[str], list[str], dict[str, int]]:
    """Compare complete displayed values and scientific figure geometry/pixels."""
    fixed_path = root / "data/reference/published_values.json"
    fixed_bytes = fixed_path.read_bytes()
    if hashlib.sha256(fixed_bytes).hexdigest() != baseline["values_sha256"]:
        message = "fixed published_values.json identity changed"
        raise ValueError(message)
    fixed = json.loads(fixed_bytes)
    current = {}
    failures, diagnostics = [], []
    for relative, expected in baseline["fragments"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"{relative}: missing manuscript fragment")
            continue
        text = path.read_text(encoding="utf-8")
        current[relative] = text
        failures.extend(fragment_differences(relative, text, expected))
        if raw_diagnostic and NUMBER.findall(
            uncomment(fixed[relative])
        ) != NUMBER.findall(uncomment(text)):
            diagnostics.append(
                f"{relative}: raw numeric tokens differ (diagnostic only)"
            )
    catalogs = _catalogs(current)
    display_count = 0
    for relative, expected in baseline["displays"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"{relative}: missing manuscript display consumer")
            continue
        actual, _ = display_sites(path.read_text(encoding="utf-8"), relative)
        actual = [
            site
            for site in actual
            if not site.expression.startswith(r"\ppvalue{lean-disclosure-")
        ]
        wanted = [_display_record(row) for row in expected]
        if actual != wanted:
            failures.append(
                f"{relative}: published numeric call sites or formatting changed"
            )
            continue
        for site, record in zip(actual, expected, strict=True):
            display_count += 1
            try:
                value = rendered_value(site, catalogs[_paper(relative)])
            except (KeyError, ValueError, ArithmeticError) as error:
                failures.append(f"{relative}: display #{site.ordinal + 1}: {error}")
                continue
            if value != record["expected"]:
                failures.append(
                    f"{relative}: display #{site.ordinal + 1} {site.expression!r}: "
                    f"{record['expected']!r} -> {value!r} "
                    f"({site.mode}, precision={site.precision}, "
                    f"scientific={site.scientific})"
                )
    for relative, expected in baseline["figures"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"{relative}: missing scientific figure")
            continue
        try:
            differences = _figure_differences(path, expected)
            failures.extend(f"{relative}: {difference}" for difference in differences)
        except (ValueError, OSError) as error:
            failures.append(f"{relative}: figure validation failed: {error}")
    return (
        failures,
        diagnostics,
        {
            "fragments": len(current),
            "displays": display_count,
            "figures": len(baseline["figures"]),
        },
    )


def _figure_differences(path: Path, expected: dict) -> list[str]:
    if path.suffix == ".pgf":
        actual = pgf_signature(
            path.read_text(encoding="utf-8"), schematic=expected["schematic"]
        )
        return plot_differences(expected["signature"], actual)
    actual = raster_signature(path.read_bytes())
    if actual == expected["signature"]:
        return []
    return ["decoded scientific raster pixels or dimensions changed"]


def _load_baseline(path: Path) -> dict:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("schema") != 1:
        message = "unsupported published-results baseline schema"
        raise ValueError(message)
    return baseline


def main() -> None:
    """Validate the four papers at their published numeric and figure precision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument(
        "--baseline", type=Path, help="Override the fixed results baseline location."
    )
    parser.add_argument(
        "--raw-diagnostic",
        action="store_true",
        help=(
            "Report stored-token differences diagnostically; "
            "unprinted digits do not cause failures."
        ),
    )
    options = parser.parse_args()
    baseline_path = (
        options.baseline or options.root / "data/reference/published_results.json"
    )
    try:
        baseline = _load_baseline(baseline_path)
        failures, diagnostics, counts = check_results(
            options.root, baseline, raw_diagnostic=options.raw_diagnostic
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Published-results comparison failed: {error}\n")
    for diagnostic in diagnostics:
        print(diagnostic, file=sys.stderr)  # noqa: T201 - CLI diagnostic
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")
    print(  # noqa: T201 - CLI result
        f"Published-precision comparison passed: {counts['displays']} numeric "
        f"displays, {counts['fragments']} fragments, {counts['figures']} figures"
    )


if __name__ == "__main__":
    main()
