#!/usr/bin/env python3
"""Deterministic vector preview renderer for the manuscript Manim series.

The source scenes in ../scenes target Manim Community Edition 0.21.x.  This
renderer exists so the project can ship with reference MP4s even in a build
sandbox where the native Manim stack is unavailable.  It deliberately mirrors
Manim's 16:9 coordinate layout and visual pacing, but it is not a replacement
for Manim.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from matplotlib.mathtext import math_to_image
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "videos"
THUMB_DIR = ROOT / "thumbnails"
sys.path.insert(0, str(ROOT))

from palette import (
    BG,
    BLUE,
    CYAN,
    FG,
    GREEN,
    GRID,
    MUTED,
    ORANGE,
    PANEL,
    PURPLE,
    RED,
    YELLOW,
)

W, H = 1280, 720
FPS = 20
DURATION = 7.5

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Manim-like coordinate map: x in [-7.111, 7.111], y in [-4, 4].
SX = W / (2 * 7.111)
SY = H / 8.0


def rgba(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    r, g, b = ImageColor.getrgb(color)
    return (r, g, b, max(0, min(255, int(round(alpha * 255)))))


def pxy(x: float, y: float) -> tuple[int, int]:
    return (int(round(W / 2 + x * SX)), int(round(H / 2 - y * SY)))


def sc(v: float) -> int:
    return int(round(v * min(SX, SY)))


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return a if x < a else b if x > b else x


def smooth(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x: float) -> float:
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def phase(
    t: float, start: float, end: float, easing: Callable[[float], float] = smooth
) -> float:
    if end <= start:
        return 1.0 if t >= end else 0.0
    return easing((t - start) / (end - start))


def lerp(a, b, u: float):
    return a + (b - a) * u


def font(size: int, bold: bool = False, mono: bool = False):
    path = FONT_MONO if mono else FONT_BOLD if bold else FONT
    if Path(path).is_file():
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def base_frame() -> Image.Image:
    im = Image.new("RGBA", (W, H), rgba(BG))
    d = ImageDraw.Draw(im, "RGBA")
    # Very subtle coordinate grid for depth.
    for x in range(0, W, 80):
        d.line((x, 0, x, H), fill=rgba(GRID, 0.12), width=1)
    for y in range(0, H, 80):
        d.line((0, y, W, y), fill=rgba(GRID, 0.12), width=1)
    d.rectangle((0, 0, W, H), outline=rgba("#27445F", 0.25), width=2)
    return im


def draw_text(
    im: Image.Image,
    xy: tuple[int, int],
    text: str,
    size: int = 32,
    color: str = FG,
    alpha: float = 1.0,
    bold: bool = False,
    anchor: str = "mm",
    mono: bool = False,
    stroke: int = 0,
) -> None:
    if alpha <= 0:
        return
    d = ImageDraw.Draw(im, "RGBA")
    d.text(
        xy,
        text,
        font=font(size, bold=bold, mono=mono),
        fill=rgba(color, alpha),
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=rgba(BG, alpha),
    )


def text_bbox(
    text: str, size: int, bold: bool = False, mono: bool = False
) -> tuple[int, int]:
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    b = d.textbbox((0, 0), text, font=font(size, bold=bold, mono=mono))
    return b[2] - b[0], b[3] - b[1]


@functools.lru_cache(maxsize=256)
def math_image(tex: str, size: int = 34, color: str = FG) -> Image.Image:
    # Matplotlib mathtext is a compact TeX subset. Normalize a few LaTeX
    # commands used by the Manim source files to their mathtext equivalents.
    tex = (
        tex.replace(r"\tfrac", r"\frac")
        .replace(r"\dfrac", r"\frac")
        .replace(r"\bigl", "")
        .replace(r"\bigr", "")
        .replace(r"\Bigl", "")
        .replace(r"\Bigr", "")
        .replace(r"\!", "")
        .replace(r"\frac12", r"\frac{1}{2}")
        .replace(r"\frac1M", r"\frac{1}{M}")
        .replace(r"\frac\lambda2", r"\frac{\lambda}{2}")
    )
    expr = tex if tex.startswith("$") else f"${tex}$"
    buf = BytesIO()
    # Render black on white, then convert luminance into a true alpha mask.
    # This avoids the opaque white rectangle emitted by math_to_image.
    math_to_image(expr, buf, dpi=180, format="png", color="black")
    buf.seek(0)
    raw = Image.open(buf).convert("RGB")
    arr = np.asarray(raw, dtype=np.uint8)
    gray = arr.mean(axis=2)
    alpha = np.clip(255.0 - gray, 0, 255).astype(np.uint8)
    target_rgb = np.array(ImageColor.getrgb(color), dtype=np.uint8)
    rgba_arr = np.empty((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba_arr[..., :3] = target_rgb
    rgba_arr[..., 3] = alpha
    img = Image.fromarray(rgba_arr, mode="RGBA")
    bbox = img.getchannel("A").getbbox()
    if bbox:
        # Preserve a small antialiasing margin.
        l, u, r, b = bbox
        bbox = (
            max(0, l - 2),
            max(0, u - 2),
            min(img.width, r + 2),
            min(img.height, b + 2),
        )
        img = img.crop(bbox)
    # The dpi/pt mapping is not exact; normalize to requested cap height.
    target_h = max(12, int(size * 1.25))
    scale = target_h / max(1, img.height)
    return img.resize(
        (max(1, int(img.width * scale)), target_h), Image.Resampling.LANCZOS
    )


def paste_math(
    im: Image.Image,
    xy: tuple[int, int],
    tex: str,
    size: int = 34,
    color: str = FG,
    alpha: float = 1.0,
    anchor: str = "mm",
) -> tuple[int, int]:
    if alpha <= 0:
        return (0, 0)
    m = math_image(tex, size, color).copy()
    if alpha < 1:
        a = m.getchannel("A").point(lambda q: int(q * alpha))
        m.putalpha(a)
    x, y = xy
    if anchor == "mm":
        pos = (x - m.width // 2, y - m.height // 2)
    elif anchor == "lm":
        pos = (x, y - m.height // 2)
    elif anchor == "rm":
        pos = (x - m.width, y - m.height // 2)
    elif anchor == "ma":
        pos = (x - m.width // 2, y)
    elif anchor == "md":
        pos = (x - m.width // 2, y - m.height)
    else:
        pos = (x, y)
    im.alpha_composite(m, pos)
    return (m.width, m.height)


def rounded_rect(
    im: Image.Image,
    box: tuple[int, int, int, int],
    fill: str = PANEL,
    outline: str = GRID,
    alpha: float = 1.0,
    width: int = 2,
    radius: int = 18,
) -> None:
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle(
        box,
        radius=radius,
        fill=rgba(fill, 0.82 * alpha),
        outline=rgba(outline, alpha),
        width=width,
    )


def panel_xy(
    im: Image.Image, cx: float, cy: float, ww: float, hh: float, alpha: float = 1.0
) -> tuple[int, int, int, int]:
    x, y = pxy(cx, cy)
    w, h = sc(ww), sc(hh)
    box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
    rounded_rect(im, box, alpha=alpha, radius=sc(0.16))
    return box


def line(
    im: Image.Image,
    a: tuple[float, float],
    b: tuple[float, float],
    color: str = FG,
    width: int = 3,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0:
        return
    ImageDraw.Draw(im, "RGBA").line(
        (pxy(*a), pxy(*b)), fill=rgba(color, alpha), width=width
    )


def dashed_line(
    im: Image.Image,
    a: tuple[float, float],
    b: tuple[float, float],
    color: str = FG,
    width: int = 2,
    alpha: float = 1.0,
    dash: int = 10,
    gap: int = 7,
) -> None:
    if alpha <= 0:
        return
    x1, y1 = pxy(*a)
    x2, y2 = pxy(*b)
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    d = ImageDraw.Draw(im, "RGBA")
    s = 0.0
    while s < dist:
        e = min(dist, s + dash)
        d.line(
            (int(x1 + ux * s), int(y1 + uy * s), int(x1 + ux * e), int(y1 + uy * e)),
            fill=rgba(color, alpha),
            width=width,
        )
        s += dash + gap


def arrow(
    im: Image.Image,
    a: tuple[float, float],
    b: tuple[float, float],
    color: str = FG,
    width: int = 4,
    alpha: float = 1.0,
    head: int = 13,
    double: bool = False,
) -> None:
    if alpha <= 0:
        return
    x1, y1 = pxy(*a)
    x2, y2 = pxy(*b)
    d = ImageDraw.Draw(im, "RGBA")
    d.line((x1, y1, x2, y2), fill=rgba(color, alpha), width=width)
    ang = math.atan2(y2 - y1, x2 - x1)

    def head_at(x, y, angle):
        pts = [
            (x, y),
            (x - head * math.cos(angle - 0.55), y - head * math.sin(angle - 0.55)),
            (x - head * math.cos(angle + 0.55), y - head * math.sin(angle + 0.55)),
        ]
        d.polygon(pts, fill=rgba(color, alpha))

    head_at(x2, y2, ang)
    if double:
        head_at(x1, y1, ang + math.pi)


def curve_arrow(
    im: Image.Image,
    a: tuple[float, float],
    b: tuple[float, float],
    bend: float,
    color: str,
    width: int = 4,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0:
        return
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)
    mid = (a_np + b_np) / 2
    v = b_np - a_np
    n = np.array([-v[1], v[0]])
    if np.linalg.norm(n) > 0:
        n /= np.linalg.norm(n)
    control = mid + bend * n
    pts = []
    for u in np.linspace(0, 1, 30):
        q = (1 - u) ** 2 * a_np + 2 * (1 - u) * u * control + u**2 * b_np
        pts.append(pxy(float(q[0]), float(q[1])))
    d = ImageDraw.Draw(im, "RGBA")
    d.line(pts, fill=rgba(color, alpha), width=width, joint="curve")
    # Arrow head on final tangent.
    p0 = np.array(pts[-2], dtype=float)
    p1 = np.array(pts[-1], dtype=float)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    head = 12 + width
    hpts = [
        (p1[0], p1[1]),
        (p1[0] - head * math.cos(ang - 0.55), p1[1] - head * math.sin(ang - 0.55)),
        (p1[0] - head * math.cos(ang + 0.55), p1[1] - head * math.sin(ang + 0.55)),
    ]
    d.polygon(hpts, fill=rgba(color, alpha))


def dot(
    im: Image.Image,
    xy: tuple[float, float],
    color: str,
    r: float = 0.09,
    alpha: float = 1.0,
    outline: str | None = None,
) -> None:
    if alpha <= 0:
        return
    x, y = pxy(*xy)
    rr = sc(r)
    d = ImageDraw.Draw(im, "RGBA")
    d.ellipse(
        (x - rr, y - rr, x + rr, y + rr),
        fill=rgba(color, alpha),
        outline=rgba(outline or color, alpha),
        width=1,
    )


def circle(
    im: Image.Image,
    xy: tuple[float, float],
    r: float,
    color: str,
    alpha: float = 1.0,
    width: int = 3,
    fill_alpha: float = 0.0,
) -> None:
    x, y = pxy(*xy)
    rr = sc(r)
    ImageDraw.Draw(im, "RGBA").ellipse(
        (x - rr, y - rr, x + rr, y + rr),
        fill=rgba(color, fill_alpha * alpha) if fill_alpha else None,
        outline=rgba(color, alpha),
        width=width,
    )


def cloud(
    im: Image.Image,
    pts: Sequence[tuple[float, float]],
    color: str,
    alpha: float = 1.0,
    r: float = 0.075,
    centers_from: tuple[float, float] | None = None,
    u: float = 1.0,
) -> None:
    for x, y in pts:
        if centers_from is not None:
            x = lerp(centers_from[0], x, u)
            y = lerp(centers_from[1], y, u)
        dot(im, (x, y), color, r=r, alpha=alpha)


def cross_marker(
    im: Image.Image,
    xy: tuple[float, float],
    color: str = GREEN,
    r: float = 0.12,
    alpha: float = 1.0,
    width: int = 3,
) -> None:
    x, y = xy
    line(im, (x - r, y), (x + r, y), color, width, alpha)
    line(im, (x, y - r), (x, y + r), color, width, alpha)


def brace_h(
    im: Image.Image, x1: float, x2: float, y: float, color: str, alpha: float = 1.0
) -> None:
    # Simple bracket rather than typographic brace.
    line(im, (x1, y), (x2, y), color, 3, alpha)
    line(im, (x1, y), (x1, y + 0.12), color, 3, alpha)
    line(im, (x2, y), (x2, y + 0.12), color, 3, alpha)


def title(
    im: Image.Image, n: str, heading: str, subtitle: str, alpha: float = 1.0
) -> None:
    if alpha <= 0:
        return
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0 = 44, 32
    d.rounded_rectangle((x0, y0, x0 + 64, y0 + 44), radius=10, fill=rgba(BLUE, alpha))
    draw_text(im, (x0 + 32, y0 + 22), n, 24, BG, alpha, bold=True)
    draw_text(im, (x0 + 83, y0 + 16), heading, 34, FG, alpha, bold=True, anchor="la")
    draw_text(im, (x0 + 83, y0 + 51), subtitle, 20, MUTED, alpha, anchor="la")
    d.line((44, 102, W - 44, 102), fill=rgba(GRID, 0.9 * alpha), width=2)


def footer(im: Image.Image, text: str, alpha: float = 1.0) -> None:
    draw_text(im, (W // 2, H - 22), text, 16, MUTED, alpha, anchor="mm")


def pill(
    im: Image.Image,
    xy: tuple[float, float],
    text: str,
    color: str,
    alpha: float = 1.0,
    text_color: str = BG,
    size: int = 20,
) -> None:
    x, y = pxy(*xy)
    tw, th = text_bbox(text, size, bold=True)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle(
        (x - tw // 2 - 14, y - th // 2 - 8, x + tw // 2 + 14, y + th // 2 + 8),
        radius=15,
        fill=rgba(color, alpha),
    )
    draw_text(im, (x, y), text, size, text_color, alpha, bold=True)


def matrix_heatmap(
    im: Image.Image,
    cx: float,
    cy: float,
    vals: np.ndarray,
    size: float,
    alpha: float = 1.0,
    symmetric: bool = False,
) -> None:
    n = vals.shape[0]
    px, py = pxy(cx, cy)
    s = sc(size)
    cell = s / n
    d = ImageDraw.Draw(im, "RGBA")
    for i in range(n):
        for j in range(n):
            v = float(vals[i, j])
            # Blend dark panel to blue/yellow based on v.
            c1 = np.array(ImageColor.getrgb(BLUE))
            c2 = np.array(ImageColor.getrgb(YELLOW))
            rgb = (c1 * (1 - v) + c2 * v).astype(int)
            x0 = px - s / 2 + j * cell
            y0 = py - s / 2 + i * cell
            d.rectangle(
                (int(x0), int(y0), int(x0 + cell - 1), int(y0 + cell - 1)),
                fill=tuple(rgb.tolist()) + (int(220 * alpha),),
            )
    d.rectangle(
        (px - s // 2, py - s // 2, px + s // 2, py + s // 2),
        outline=rgba(FG, alpha),
        width=2,
    )
    if symmetric:
        d.line(
            (px - s // 2, py - s // 2, px + s // 2, py + s // 2),
            fill=rgba(FG, 0.7 * alpha),
            width=2,
        )


# ---------- Scene functions ----------


def scene01(t: float) -> Image.Image:
    im = base_frame()
    a = phase(t, 0, 0.65)
    title(
        im,
        "01",
        "A firm as a probability law",
        "Retaining spread, multimodality and internal composition",
        a,
    )
    panels = phase(t, 0.55, 1.35)
    panel_xy(im, -3.35, -0.15, 5.6, 3.75, panels)
    panel_xy(im, 3.35, -0.15, 5.6, 3.75, panels)
    draw_text(im, pxy(-3.35, 1.35), "Firm A", 25, BLUE, panels, bold=True)
    draw_text(im, pxy(3.35, 1.35), "Firm B", 25, YELLOW, panels, bold=True)
    point_a = phase(t, 1.15, 1.8) * (1 - phase(t, 3.0, 4.3))
    dot(im, (-3.35, -0.05), BLUE, 0.14, point_a)
    dot(im, (3.35, -0.05), YELLOW, 0.14, point_a)
    draw_text(
        im,
        pxy(0, -2.28),
        "Point summaries: identical centroids",
        24,
        RED,
        point_a,
        bold=True,
    )

    expand = phase(t, 3.0, 4.45)
    a_pts = [
        (-3.75, -0.55),
        (-3.55, 0.05),
        (-3.5, -0.75),
        (-3.2, 0.25),
        (-3.0, -0.35),
        (-3.62, 0.58),
        (-3.08, 0.72),
        (-2.78, 0.12),
        (-2.75, -0.68),
        (-3.25, -0.08),
    ]
    b_pts = [
        (2.35, -0.58),
        (2.55, -0.12),
        (2.68, 0.38),
        (2.48, 0.72),
        (2.72, -0.82),
        (3.88, -0.56),
        (4.08, -0.08),
        (4.22, 0.4),
        (4.05, 0.78),
        (3.78, -0.86),
    ]
    cloud(im, a_pts, BLUE, alpha=expand, r=0.085, centers_from=(-3.35, -0.05), u=expand)
    cloud(
        im, b_pts, YELLOW, alpha=expand, r=0.085, centers_from=(3.35, -0.05), u=expand
    )
    labels = phase(t, 4.25, 5.25)
    cross_marker(im, (-3.35, -0.05), GREEN, 0.12, labels)
    cross_marker(im, (3.35, -0.05), GREEN, 0.12, labels)
    draw_text(im, pxy(-3.35, -1.55), "one broad mode", 21, BLUE, labels)
    draw_text(im, pxy(3.35, -1.55), "two separated modes", 21, YELLOW, labels)
    draw_text(
        im,
        pxy(0, -2.22),
        "Same mean, different composition",
        25,
        GREEN,
        labels,
        bold=True,
    )
    eq = phase(t, 5.15, 6.25)
    paste_math(
        im,
        pxy(0, -2.92),
        r"C_i\in\mathcal{P}_2(\Omega),\qquad C_i=\delta_{x_i}\ \mathrm{is\ the\ point\ special\ case}",
        27,
        FG,
        eq,
    )
    footer(
        im,
        "Thesis §1.2.1 — the distribution as a modelling primitive",
        phase(t, 5.7, 6.4),
    )
    return im


def scene02(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "02",
        "Couplings and quadratic transport",
        "Which observations from two distributions occur together?",
        phase(t, 0, 0.65),
    )
    pa = phase(t, 0.55, 1.25)
    panel_xy(im, 0, -0.12, 12.2, 4.25, pa)
    ysL = [1.02, 0.35, -0.35, -1.02]
    ysR = [0.9, -0.9, 0.18, -0.18]
    ptsA = [(-4.5, y) for y in ysL]
    ptsB = [(4.5, y) for y in ysR]
    pts_alpha = phase(t, 1.0, 1.8)
    cloud(im, ptsA, BLUE, pts_alpha, 0.1)
    cloud(im, ptsB, YELLOW, pts_alpha, 0.1)
    paste_math(im, pxy(-5.2, 0), r"C_i", 31, BLUE, pts_alpha)
    paste_math(im, pxy(5.2, 0), r"C_j", 31, YELLOW, pts_alpha)
    bad = [1, 0, 3, 2]
    good = [0, 2, 3, 1]
    bad_a = phase(t, 1.65, 2.6) * (1 - phase(t, 3.5, 4.5))
    for k, j in enumerate(bad):
        line(im, ptsA[k], ptsB[j], RED, 4, bad_a * 0.85)
    draw_text(
        im,
        pxy(0, 1.55),
        "One feasible coupling — large displacement",
        23,
        RED,
        bad_a,
        bold=True,
    )
    u = phase(t, 3.35, 4.85)
    good_a = phase(t, 3.35, 4.0)
    for k in range(4):
        yend = lerp(ptsB[bad[k]][1], ptsB[good[k]][1], u)
        line(im, ptsA[k], (ptsB[good[k]][0], yend), GREEN, 4, good_a * 0.9)
    draw_text(
        im,
        pxy(0, 1.55),
        "Optimal coupling — minimum average squared displacement",
        23,
        GREEN,
        phase(t, 4.05, 4.9),
        bold=True,
    )
    draw_text(
        im,
        pxy(0, -1.7),
        "A coupling π is a feasible joint assignment with the prescribed marginals",
        20,
        MUTED,
        phase(t, 4.5, 5.4),
    )
    eq = phase(t, 5.0, 6.15)
    paste_math(
        im,
        pxy(0, -2.52),
        r"W_2^2(C_i,C_j)=\inf_{\pi\in\Pi(C_i,C_j)}\ \mathbb{E}_{\pi}[d(X,Y)^2]",
        31,
        FG,
        eq,
    )
    footer(
        im,
        "Thesis Eq. (1.2) — nearby moves are cheaper than distant moves",
        phase(t, 5.6, 6.4),
    )
    return im


def scene03(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "03",
        "Polarization: distance becomes covariance",
        "The algebraic bridge from transport geometry to second moments",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.25)
    panel_xy(im, -3.4, -0.15, 6.0, 4.15, pp)
    panel_xy(im, 3.55, -0.15, 5.5, 4.15, pp)
    origin = (-4.5, -0.65)
    line(im, (-6.1, -0.65), (-1.2, -0.65), MUTED, 2, pp)
    line(im, origin, (-4.5, 1.25), MUTED, 2, pp)
    zi_end = (-2.55, -0.25)
    arrow(im, origin, zi_end, BLUE, 6, phase(t, 1.0, 1.8))
    paste_math(im, pxy(-2.3, -0.2), r"z_i", 31, BLUE, phase(t, 1.2, 1.9))
    rot = phase(t, 2.3, 5.1)
    ang = math.radians(100 - 67 * rot)
    length = 2.05
    zj_end = (origin[0] + length * math.cos(ang), origin[1] + length * math.sin(ang))
    arrow(im, origin, zj_end, YELLOW, 6, phase(t, 1.35, 2.05))
    paste_math(
        im,
        pxy(zj_end[0] + 0.18, zj_end[1] + 0.18),
        r"z_j",
        31,
        YELLOW,
        phase(t, 1.5, 2.1),
    )
    dashed_line(
        im, zi_end, zj_end, RED if rot < 0.55 else GREEN, 3, phase(t, 1.75, 2.3), 10, 7
    )
    paste_math(
        im,
        pxy(3.55, 0.9),
        r"\langle z_i,z_j\rangle=\tfrac12\!\left(\|z_i\|^2+\|z_j\|^2-\|z_i-z_j\|^2\right)",
        27,
        FG,
        phase(t, 1.25, 2.2),
    )
    draw_text(
        im, pxy(3.55, 0.18), "Hold both magnitudes fixed", 21, MUTED, phase(t, 2.0, 2.8)
    )
    # Dynamic bars.
    d2 = (zi_end[0] - zj_end[0]) ** 2 + (zi_end[1] - zj_end[1]) ** 2
    ip = (zi_end[0] - origin[0]) * (zj_end[0] - origin[0]) + (zi_end[1] - origin[1]) * (
        zj_end[1] - origin[1]
    )
    maxd = 8.0
    maxip = 4.2
    draw_text(
        im,
        pxy(2.2, -0.55),
        "squared distance",
        19,
        RED,
        phase(t, 2.2, 3.0),
        anchor="lm",
    )
    draw_text(
        im,
        pxy(2.2, -1.15),
        "inner product",
        19,
        PURPLE,
        phase(t, 2.2, 3.0),
        anchor="lm",
    )
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0 = pxy(3.5, -0.55)
    d.rounded_rectangle(
        (x0, y0 - 10, x0 + 260, y0 + 10), radius=8, fill=rgba(GRID, 0.8)
    )
    d.rounded_rectangle(
        (x0, y0 - 10, x0 + int(260 * clamp(d2 / maxd)), y0 + 10),
        radius=8,
        fill=rgba(RED, 0.9),
    )
    x1, y1 = pxy(3.5, -1.15)
    d.rounded_rectangle(
        (x1, y1 - 10, x1 + 260, y1 + 10), radius=8, fill=rgba(GRID, 0.8)
    )
    d.rounded_rectangle(
        (x1, y1 - 10, x1 + int(260 * clamp((ip + 1) / maxip)), y1 + 10),
        radius=8,
        fill=rgba(PURPLE, 0.9),
    )
    draw_text(
        im,
        pxy(3.55, -1.72),
        "smaller distance  ⇔  larger alignment",
        22,
        GREEN,
        phase(t, 4.6, 5.5),
        bold=True,
    )
    eq = phase(t, 5.25, 6.35)
    paste_math(
        im,
        pxy(0, -2.72),
        r"\min_{\pi}\mathbb{E}_{\pi}\|Z_i-Z_j\|^2\quad\Longleftrightarrow\quad\max_{\pi}\mathbb{E}_{\pi}\langle Z_i,Z_j\rangle",
        30,
        FG,
        eq,
    )
    footer(
        im,
        "Thesis Eq. (1.3), Lemma 3.1 — minimizing quadratic transport maximizes attainable covariance",
        phase(t, 5.8, 6.5),
    )
    return im


def mini_matching(
    im: Image.Image, cx: float, cy: float, perm: Sequence[int], color: str, alpha: float
):
    ys = [cy + 0.45, cy, cy - 0.45]
    xl = cx - 0.72
    xr = cx + 0.72
    for y in ys:
        dot(im, (xl, y), BLUE, 0.045, alpha)
        dot(im, (xr, y), YELLOW, 0.045, alpha)
    for k, j in enumerate(perm):
        line(im, (xl, ys[k]), (xr, ys[j]), color, 2, alpha)


def scene04(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "04",
        "A sharp covariance envelope",
        "Vary the coupling while keeping both exposure laws fixed",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.2)
    panel_xy(im, 0, 0.45, 12.2, 3.1, pp)
    cards = [
        (-3.0, "reflected floor", RED, [2, 0, 1]),
        (0, "realized coupling", CYAN, [0, 2, 1]),
        (3.0, "W₂ ceiling", GREEN, [0, 1, 2]),
    ]
    for idx, (cx, txt, col, perm) in enumerate(cards):
        a = phase(t, 1.0 + idx * 0.3, 1.8 + idx * 0.3)
        panel_xy(im, cx, 0.45, 2.5, 1.95, a)
        mini_matching(im, cx, 0.55, perm, col, a)
        draw_text(im, pxy(cx, -0.38), txt, 18, col, a, bold=True)
    # Interval.
    a = phase(t, 2.4, 3.25)
    line(im, (-4.0, -1.15), (4.0, -1.15), MUTED, 3, a)
    line(im, (-3.0, -1.15), (3.0, -1.15), PURPLE, 8, a)
    for x, col, tex, up in [
        (-3, RED, r"\kappa^-_{ij}", False),
        (0, CYAN, r"\kappa_{\pi}", True),
        (3, GREEN, r"\bar\kappa_{ij}", False),
    ]:
        dot(im, (x, -1.15), col, 0.10, a)
        paste_math(im, pxy(x, -0.75 if up else -1.58), tex, 27, col, a)
    # transport excess bracket
    gap = phase(t, 3.25, 4.15)
    brace_h(im, 0, 3, -0.62, YELLOW, gap)
    paste_math(im, pxy(1.5, -0.36), r"\tfrac12\Delta_{\pi}", 25, YELLOW, gap)
    draw_text(
        im,
        pxy(0, -2.0),
        "The marginals define an attainable interval — not the realized point",
        22,
        MUTED,
        phase(t, 3.7, 4.7),
    )
    eq = phase(t, 4.55, 5.75)
    paste_math(
        im,
        pxy(0, -2.65),
        r"\kappa_{\pi}=\bar\kappa_{ij}-\tfrac12\Delta_{\pi},\qquad \Delta_{\pi}\geq 0",
        34,
        FG,
        eq,
    )
    footer(
        im,
        "Thesis Theorems 3.1–3.2 — reflection supplies the lower endpoint",
        phase(t, 5.2, 6.0),
    )
    return im


def scene05(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "05",
        "Transmission, distortion and slack",
        "Why information distance is not silently equated with risk distance",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.25)
    panel_xy(im, -3.8, -0.1, 5.0, 3.85, pp)
    panel_xy(im, 3.8, -0.1, 5.0, 3.85, pp)
    draw_text(
        im, pxy(-3.8, 1.45), "observable information space", 22, BLUE, pp, bold=True
    )
    draw_text(im, pxy(3.8, 1.45), "latent factor-risk space", 22, ORANGE, pp, bold=True)
    c1 = [(-5.05, 0.55), (-4.72, 0.2), (-4.9, -0.22), (-4.58, -0.62)]
    c2 = [(-3.02, 0.55), (-2.7, 0.2), (-2.85, -0.23), (-2.53, -0.62)]
    aa = phase(t, 1.0, 1.8)
    cloud(im, c1, BLUE, aa, 0.085)
    cloud(im, c2, CYAN, aa, 0.085)
    arrow(im, (-4.6, -1.1), (-2.75, -1.1), GREEN, 3, phase(t, 1.5, 2.1), 12, True)
    paste_math(
        im, pxy(-3.67, -0.88), r"G_{ij}=W_2(C_i,C_j)", 25, GREEN, phase(t, 1.55, 2.2)
    )
    map_a = phase(t, 1.9, 2.7)
    arrow(im, (-1.05, 0), (1.05, 0), YELLOW, 5, map_a, 18)
    paste_math(im, pxy(0, 0.45), r"T(x,U)", 30, YELLOW, map_a)
    draw_text(im, pxy(0, 0.05), "common carrier", 18, MUTED, map_a)
    p1 = [(2.55, 0.52), (2.88, 0.18), (2.6, -0.28), (3.05, -0.62)]
    p2 = [(4.43, 0.52), (4.86, 0.22), (4.57, -0.25), (5.06, -0.58)]
    ab = phase(t, 2.35, 3.35)
    cloud(im, p1, ORANGE, ab, 0.085, centers_from=(2.8, 0), u=ab)
    cloud(im, p2, RED, ab, 0.085, centers_from=(4.8, 0), u=ab)
    circle(im, (2.8, 0), 0.5, YELLOW, phase(t, 2.8, 3.55), 2)
    circle(im, (4.8, 0), 0.5, YELLOW, phase(t, 2.8, 3.55), 2)
    paste_math(im, pxy(2.8, -0.9), r"\tau_i", 24, YELLOW, phase(t, 2.9, 3.6))
    paste_math(im, pxy(4.8, -0.9), r"\tau_j", 24, YELLOW, phase(t, 2.9, 3.6))
    arrow(im, (3.15, -1.15), (4.45, -1.15), CYAN, 3, phase(t, 3.2, 4.0), 12, True)
    paste_math(
        im, pxy(3.8, -0.9), r"W_{2,\Gamma}(P_i,P_j)", 23, CYAN, phase(t, 3.25, 4.05)
    )
    # Distance interval widens as sensitivity relaxes.
    sens = phase(t, 4.0, 5.6)
    lower = 2.3 * (1 - 0.55 * sens)
    upper = 2.3 * (1 + 0.45 * sens)
    line(im, (-upper, -2.0), (upper, -2.0), MUTED, 2, phase(t, 3.8, 4.4))
    line(im, (-lower, -2.0), (lower, -2.0), GREEN, 8, phase(t, 3.8, 4.4))
    paste_math(
        im,
        pxy(0, -1.72),
        r"\mathrm{admissible\ latent\ distance}",
        22,
        MUTED,
        phase(t, 3.9, 4.5),
    )
    paste_math(
        im,
        pxy(0, -2.58),
        r"(L^{-1}G_{ij}-\tau_i-\tau_j)_+\leq W_{2,\Gamma}(P_i,P_j)\leq LG_{ij}+\tau_i+\tau_j",
        27,
        FG,
        phase(t, 4.6, 5.7),
    )
    footer(
        im,
        "Thesis Corollary 3.1 — L and τ are declared sensitivity inputs, not fitted return parameters",
        phase(t, 5.25, 6.0),
    )
    return im


def scene06(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "06",
        "Target-anchored barycentric reconstruction",
        "Transport establishes correspondence; a simplex problem chooses peer coordinates",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.2)
    panel_xy(im, -4.8, -0.1, 3.05, 3.95, pp)
    panel_xy(im, -0.8, -0.1, 4.25, 3.95, pp)
    panel_xy(im, 3.8, -0.1, 4.25, 3.95, pp)
    draw_text(im, pxy(-4.8, 1.45), "A. fixed target", 21, BLUE, pp, bold=True)
    draw_text(im, pxy(-0.8, 1.45), "B. aligned peers", 21, CYAN, pp, bold=True)
    draw_text(im, pxy(3.8, 1.45), "C. reconstruction", 21, GREEN, pp, bold=True)
    target = [
        (-5.25, 0.72),
        (-4.55, 0.55),
        (-5.08, -0.03),
        (-4.45, -0.45),
        (-5.15, -0.9),
    ]
    ta = phase(t, 1.0, 1.8)
    cloud(im, target, BLUE, ta, 0.09)
    paste_math(im, pxy(-4.8, -1.45), r"\widehat C_i", 28, BLUE, ta)
    peers = [
        [(-2.0, 0.72), (-1.65, 0.55), (-1.9, -0.03), (-1.55, -0.45), (-2.0, -0.9)],
        [(-0.8, 0.72), (-0.45, 0.55), (-0.7, -0.03), (-0.35, -0.45), (-0.8, -0.9)],
        [(0.4, 0.72), (0.75, 0.55), (0.5, -0.03), (0.85, -0.45), (0.4, -0.9)],
    ]
    cols = [YELLOW, ORANGE, PURPLE]
    pa = phase(t, 1.45, 2.35)
    for ps, col in zip(peers, cols):
        cloud(im, ps, col, pa, 0.065)
    for k, col in enumerate(cols):
        paste_math(im, pxy(-2.0 + 1.2 * k, -1.42), rf"j_{k + 1}", 23, col, pa)
    match = phase(t, 2.1, 3.1) * (1 - phase(t, 4.0, 4.7))
    # Visual target-to-peer assignments: normalize target into middle box for lines.
    target_mid = [
        (-2.65, 0.72),
        (-2.65, 0.35),
        (-2.65, -0.03),
        (-2.65, -0.42),
        (-2.65, -0.82),
    ]
    for m in range(5):
        for ps, col in zip(peers, cols):
            dashed_line(im, target_mid[m], ps[m], col, 1, match, 7, 6)
    draw_text(im, pxy(-0.8, -1.75), "one matching problem per peer", 18, MUTED, match)
    weights = [0.42, 0.35, 0.23]
    ra = phase(t, 3.45, 4.65)
    # Reconstructed green cloud and a faint target ghost.
    ghost = [(x + 8.6, y) for x, y in target]
    cloud(im, ghost, BLUE, ra * 0.28, 0.09)
    # Weighted blend of aligned peer locations shifted into the right panel.
    recon = []
    for m in range(5):
        v = sum(weights[k] * np.array(peers[k][m]) for k in range(3))
        recon.append((float(v[0] + 4.6), float(v[1])))
    cloud(im, recon, GREEN, ra, 0.09, centers_from=(3.8, 0), u=ra)
    # Weight pills.
    for k, (w, col) in enumerate(zip(weights, cols)):
        pill(
            im,
            (2.85 + 0.95 * k, -1.45),
            f"{w:.2f}",
            col,
            phase(t, 4.15 + k * 0.08, 4.8 + k * 0.08),
            FG if col == PURPLE else BG,
            18,
        )
    paste_math(
        im,
        pxy(0, -2.48),
        r"w_i\in\arg\min_{w\in\Delta_{-i}}\frac1M\sum_m\left\|x_{im}-\sum_{j\ne i}w_jy_{ijm}\right\|^2",
        28,
        FG,
        phase(t, 4.75, 5.8),
    )
    footer(
        im,
        "Thesis Eqs. (4.11)–(4.12) — the target and pairwise alignments are fixed before optimizing weights",
        phase(t, 5.25, 6.0),
    )
    return im


def scene07(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "07",
        "A symmetric metric can yield a directed field",
        "Distance measures separation; reconstruction measures conditional usefulness",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.2)
    panel_xy(im, -3.8, -0.1, 5.0, 4.0, pp)
    panel_xy(im, 3.35, -0.1, 6.1, 4.0, pp)
    draw_text(
        im, pxy(-3.8, 1.45), "symmetric pairwise geometry", 21, CYAN, pp, bold=True
    )
    draw_text(
        im, pxy(3.35, 1.45), "target-specific reconstruction", 21, GREEN, pp, bold=True
    )
    L = [(-4.75, 0.52), (-2.85, 0.52), (-3.8, -0.9)]
    cols = [BLUE, YELLOW, PURPLE]
    aa = phase(t, 1.0, 1.8)
    for p, c, n in zip(L, cols, "ABC"):
        dot(im, p, c, 0.16, aa)
        draw_text(im, pxy(p[0], p[1] + 0.32), n, 22, c, aa, bold=True)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        line(im, L[a], L[b], MUTED, 3, aa)
    paste_math(
        im, pxy(-3.8, -1.55), r"W_2(C_i,C_j)=W_2(C_j,C_i)", 27, CYAN, phase(t, 1.5, 2.2)
    )
    R = [(1.45, 0.58), (5.05, 0.72), (3.2, -1.0)]
    rb = phase(t, 2.0, 2.8)
    for p, c, n in zip(R, cols, "ABC"):
        dot(im, p, c, 0.17, rb)
        draw_text(im, pxy(p[0], p[1] + 0.35), n, 23, c, rb, bold=True)
    ar = phase(t, 2.55, 3.8)
    curve_arrow(im, R[0], R[1], -0.38, YELLOW, 5, ar)
    paste_math(im, pxy(3.18, 1.18), r"0.60", 22, YELLOW, ar)
    curve_arrow(im, R[1], R[0], -0.38, BLUE, 2, ar)
    paste_math(im, pxy(3.18, 0.22), r"0.10", 22, BLUE, ar)
    curve_arrow(im, R[0], R[2], 0.28, PURPLE, 4, ar)
    paste_math(im, pxy(2.05, -0.28), r"0.40", 22, PURPLE, ar)
    curve_arrow(im, R[2], R[1], 0.28, YELLOW, 5, ar)
    paste_math(im, pxy(4.35, -0.28), r"0.75", 22, YELLOW, ar)
    paste_math(
        im,
        pxy(3.35, -1.6),
        r"W^{\flat}_{ij}\ne W^{\flat}_{ji}",
        34,
        GREEN,
        phase(t, 3.65, 4.55),
    )
    draw_text(
        im,
        pxy(0, -2.55),
        "Direction records reconstruction relevance — not causal influence",
        24,
        MUTED,
        phase(t, 4.25, 5.2),
        bold=True,
    )
    footer(
        im,
        "Thesis §4.4 — every row is nonnegative, unit-sum and zero-diagonal",
        phase(t, 4.8, 5.55),
    )
    return im


def scene08(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "08",
        "Quadratic adjustment implies spatial closure",
        "Deriving the spatial lag at the exposure layer",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.2)
    panel_xy(im, 0, -0.1, 12.2, 4.2, pp)
    y = 0.65
    line(im, (-4.7, y), (4.7, y), MUTED, 3, pp)
    xi = (-3.8, y)
    peer = (3.8, y)
    dot(im, xi, BLUE, 0.14, phase(t, 1, 1.7))
    dot(im, peer, YELLOW, 0.14, phase(t, 1, 1.7))
    paste_math(im, pxy(-3.8, 1.05), r"\xi_i", 30, BLUE, phase(t, 1, 1.7))
    paste_math(im, pxy(3.8, 1.05), r"\sum_jW_{ij}B_j", 27, YELLOW, phase(t, 1, 1.7))
    # lambda increases, rho increases, and B moves toward peer average.
    u = phase(t, 2.0, 4.75)
    rho_v = 0.78 * u
    bx = lerp(xi[0], peer[0], rho_v)
    dot(im, (bx, y), GREEN, 0.16, phase(t, 1.65, 2.2))
    paste_math(im, pxy(bx, 0.22), r"B_i", 30, GREEN, phase(t, 1.7, 2.3))
    brace_h(im, xi[0], bx, 0.05, BLUE, phase(t, 2.1, 2.8))
    brace_h(im, bx, peer[0], -0.42, YELLOW, phase(t, 2.1, 2.8))
    draw_text(
        im,
        pxy((xi[0] + bx) / 2, -0.2),
        "stand-alone deviation",
        17,
        BLUE,
        phase(t, 2.25, 3.0),
    )
    draw_text(
        im,
        pxy((bx + peer[0]) / 2, -0.68),
        "peer misalignment",
        17,
        YELLOW,
        phase(t, 2.25, 3.0),
    )
    paste_math(
        im,
        pxy(0, -1.25),
        r"\min_a\ \tfrac12\|a-\xi_i\|^2+\tfrac\lambda2\sum_jW_{ij}\|a-B_j\|^2",
        29,
        FG,
        phase(t, 2.55, 3.55),
    )
    # rho meter
    draw_text(
        im,
        pxy(-2.6, -1.8),
        "relative adjustment",
        19,
        MUTED,
        phase(t, 3.0, 3.7),
        anchor="lm",
    )
    x0, y0 = pxy(-0.9, -1.8)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle(
        (x0, y0 - 11, x0 + 310, y0 + 11), radius=9, fill=rgba(GRID, 0.9)
    )
    d.rounded_rectangle(
        (x0, y0 - 11, x0 + int(310 * rho_v / 0.78), y0 + 11),
        radius=9,
        fill=rgba(GREEN, 0.95),
    )
    paste_math(
        im,
        pxy(2.95, -1.8),
        rf"\rho=\frac{{\lambda}}{{1+\lambda}}\approx {rho_v:.2f}",
        25,
        GREEN,
        phase(t, 3.0, 3.7),
    )
    paste_math(
        im, pxy(0, -2.55), r"B=\rho WB+(1-\rho)\xi", 37, GREEN, phase(t, 4.65, 5.55)
    )
    footer(
        im,
        "Thesis Theorems 4.1–4.2 — (1−ρ)(I−ρW)⁻¹ accumulates direct and higher-order adjustment",
        phase(t, 5.1, 5.9),
    )
    return im


def matching_card(im, cx, perms, col_left, col_right, prefixL, prefixR, alpha):
    panel_xy(im, cx, 0.15, 3.95, 3.55, alpha)
    ys = [0.85, 0.15, -0.55]
    xl = cx - 1.25
    xr = cx + 1.25
    for k, y in enumerate(ys):
        dot(im, (xl, y), col_left, 0.085, alpha)
        dot(im, (xr, y), col_right, 0.085, alpha)
        paste_math(im, pxy(xl - 0.35, y), rf"{prefixL}_{k}", 19, col_left, alpha)
        paste_math(im, pxy(xr + 0.35, y), rf"{prefixR}_{k}", 19, col_right, alpha)
    for k, j in enumerate(perms):
        line(im, (xl, ys[k]), (xr, ys[j]), GREEN, 3, alpha)


def scene09(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "09",
        "Pairwise plans need not form one joint law",
        "Portfolio aggregation adds a compatibility requirement",
        phase(t, 0, 0.65),
    )
    specs = [
        (-4.15, [1, 0, 2], BLUE, YELLOW, "A", "B", "A ↔ B optimum"),
        (0, [2, 1, 0], YELLOW, PURPLE, "B", "C", "B ↔ C optimum"),
        (4.15, [0, 2, 1], BLUE, PURPLE, "A", "C", "A ↔ C optimum"),
    ]
    for idx, (cx, perm, cl, cr, pl, pr, head) in enumerate(specs):
        aa = phase(t, 0.7 + idx * 0.22, 1.55 + idx * 0.22)
        matching_card(im, cx, perm, cl, cr, pl, pr, aa)
        draw_text(im, pxy(cx, 1.55), head, 19, FG, aa, bold=True)
    trace = phase(t, 2.6, 3.5)
    # Emphasize A0 -> B1 -> C1 against A0 -> C0.
    draw_text(
        im,
        pxy(0, -1.65),
        "Follow one atom through the first two plans:",
        20,
        MUTED,
        trace,
    )
    paste_math(
        im,
        pxy(0, -2.05),
        r"A_0\longrightarrow B_1\longrightarrow C_1",
        31,
        YELLOW,
        trace,
    )
    direct = phase(t, 3.45, 4.3)
    paste_math(
        im,
        pxy(0, -2.58),
        r"\mathrm{but\ the\ direct\ plan\ requires}\quad A_0\longrightarrow C_0",
        29,
        RED,
        direct,
    )
    # Big cross.
    ca = phase(t, 4.2, 4.75)
    line(im, (-1.1, -2.85), (1.1, -2.3), RED, 7, ca)
    line(im, (-1.1, -2.3), (1.1, -2.85), RED, 7, ca)
    conclusion = phase(t, 4.65, 5.55)
    draw_text(
        im,
        pxy(0, -3.12),
        "No single J has all three pairwise plans as marginals",
        24,
        RED,
        conclusion,
        bold=True,
    )
    footer(
        im,
        "Concrete three-point squared-Euclidean assignment example; the manuscript §5.2–§5.3 retains one coherent joint law",
        phase(t, 5.15, 5.95),
    )
    return im


def scene10(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "10",
        "Multi-firm transport dispersion",
        "One coherent coupling — equivalently, a free Wasserstein centre",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.2)
    panel_xy(im, -3.45, -0.1, 6.25, 4.15, pp)
    panel_xy(im, 3.65, -0.1, 5.65, 4.15, pp)
    draw_text(im, pxy(-3.45, 1.48), "common-coupling view", 21, CYAN, pp, bold=True)
    draw_text(im, pxy(3.65, 1.48), "free-centre view", 21, GREEN, pp, bold=True)
    centers = [(-5.2, 0.55), (-3.45, 0.55), (-1.7, 0.55)]
    cols = [BLUE, YELLOW, PURPLE]
    clouds = []
    for c, col in zip(centers, cols):
        pts = [
            (c[0] - 0.25, c[1] + 0.5),
            (c[0] + 0.18, c[1] + 0.18),
            (c[0] - 0.1, c[1] - 0.28),
            (c[0] + 0.3, c[1] - 0.65),
        ]
        clouds.append(pts)
    aa = phase(t, 1.0, 1.8)
    for i, (pts, col) in enumerate(zip(clouds, cols)):
        cloud(im, pts, col, aa, 0.08)
        paste_math(im, pxy(centers[i][0], -1.25), rf"C_{i + 1}", 24, col, aa)
    tri = phase(t, 1.75, 2.75)
    for m in range(4):
        line(im, clouds[0][m], clouds[1][m], MUTED, 2, tri * 0.8)
        line(im, clouds[1][m], clouds[2][m], MUTED, 2, tri * 0.8)
        line(im, clouds[0][m], clouds[2][m], MUTED, 2, tri * 0.8)
    draw_text(
        im,
        pxy(-3.45, -1.75),
        "one realization joins every firm at once",
        18,
        MUTED,
        phase(t, 2.25, 3.05),
    )
    rc = [(2.0, 0.75), (5.25, 0.75), (3.65, -0.85)]
    rb = phase(t, 2.3, 3.2)
    for i, (c, col) in enumerate(zip(rc, cols)):
        circle(im, c, 0.58, col, rb, 3, 0.05)
        paste_math(im, pxy(*c), rf"P_{i + 1}", 24, col, rb)
    q_u = phase(t, 3.1, 4.5)
    q = (lerp(2.45, 3.65, q_u), lerp(-0.2, 0.12, q_u))
    for c, col in zip(rc, cols):
        dashed_line(im, c, q, col, 2, phase(t, 3.0, 3.8), 9, 7)
    dot(im, q, GREEN, 0.15, phase(t, 3.0, 3.7))
    paste_math(im, pxy(q[0], q[1] + 0.35), r"Q", 30, GREEN, phase(t, 3.0, 3.7))
    paste_math(
        im,
        pxy(-3.45, -2.55),
        r"D_q(C)=\inf_{\gamma}\mathbb{E}_{\gamma}\sum_{i<j}q_iq_jd(X_i,X_j)^2",
        24,
        FG,
        phase(t, 3.9, 5.0),
    )
    paste_math(
        im,
        pxy(3.65, -2.15),
        r"D_q(P)=\inf_Q\sum_iq_iW_2^2(P_i,Q)",
        28,
        GREEN,
        phase(t, 4.1, 5.2),
    )
    draw_text(
        im,
        pxy(3.65, -2.68),
        "Free Q, fixed portfolio weights q",
        20,
        MUTED,
        phase(t, 4.5, 5.35),
        bold=True,
    )
    footer(
        im,
        "Thesis Eqs. (5.9)–(5.11) — this free-centre problem is distinct from Essay II's fixed-target reconstruction",
        phase(t, 5.0, 5.8),
    )
    return im


def scene11(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "11",
        "Portfolio variance as alignment minus dispersion",
        "Information rules out part of the worst-case risk benchmark",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.2)
    panel_xy(im, 0, -0.1, 12.2, 4.2, pp)
    origin = (-4.8, -0.75)
    cols = [BLUE, YELLOW, PURPLE, ORANGE]
    u = phase(t, 2.5, 4.4)
    a = phase(t, 1.0, 1.9) * (1 - u)
    aligned = [(-2.4, 0.55), (-2.15, 0.68), (-1.9, 0.8), (-1.65, 0.92)]
    for e, c in zip(aligned, cols):
        arrow(im, origin, e, c, 6, a, 15)
    draw_text(im, pxy(-3.25, 1.52), "perfect positive alignment", 22, RED, a, bold=True)
    # Risk bar.
    bx, by = pxy(0.7, 0.1)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle(
        (bx - 70, by - 150, bx + 70, by + 150),
        radius=16,
        fill=rgba(RED, 0.16 * a),
        outline=rgba(RED, a),
        width=3,
    )
    d.rectangle((bx - 50, by - 125, bx + 50, by + 125), fill=rgba(RED, 0.82 * a))
    draw_text(im, (bx, by - 175), "admissible risk cap", 20, FG, a, bold=True)
    paste_math(im, (bx, by), r"\sum_iq_iv_i", 25, BG, a)
    # Disperse vectors.
    spread = [(-2.2, 0.55), (-3.0, 1.35), (-2.7, -0.75), (-4.0, 1.3)]
    for e0, e1, c in zip(aligned, spread, cols):
        arrow(im, origin, (lerp(e0[0], e1[0], u), lerp(e0[1], e1[1], u)), c, 6, u, 15)
    draw_text(
        im,
        pxy(-3.25, 1.52),
        "certified separation floors",
        22,
        GREEN,
        phase(t, 3.5, 4.4),
        bold=True,
    )
    # Cut risk bar.
    cut = phase(t, 3.5, 4.75)
    cut_h = int(90 * cut)
    d.rectangle((bx - 50, by + 125 - cut_h, bx + 50, by + 125), fill=rgba(GREEN, 0.95))
    if cut > 0:
        paste_math(im, (bx, by + 125 - cut_h // 2), r"C(q)", 23, BG, cut)
    # Cover upper bar to show reduced cap.
    if cut > 0:
        d.rectangle(
            (bx - 50, by - 125, bx + 50, by + 125 - cut_h), fill=rgba(YELLOW, 0.86)
        )
    rt = phase(t, 4.0, 5.0)
    draw_text(im, pxy(4.1, 0.72), "not a covariance estimate", 21, MUTED, rt)
    draw_text(im, pxy(4.1, 0.12), "a one-sided certificate", 25, GREEN, rt, bold=True)
    draw_text(im, pxy(4.1, -0.48), "about admissible risk", 21, MUTED, rt)
    paste_math(
        im,
        pxy(0, -2.55),
        r"V_{\mathrm{sys}}(q)\leq\sum_iq_iv_i-\frac12\sum_{i,j}q_iq_j\ell_{ij}^2",
        32,
        FG,
        phase(t, 4.85, 5.85),
    )
    footer(
        im,
        "Thesis Theorem 5.1 — the certificate deducts information-certified diversification from perfect alignment",
        phase(t, 5.3, 6.1),
    )
    return im


def scene12(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "12",
        "Geometry can certify convex optimization",
        "A checkable condition on the floor matrix — no return covariance required",
        phase(t, 0, 0.65),
    )
    pp = phase(t, 0.55, 1.2)
    panel_xy(im, -3.45, -0.1, 6.0, 4.15, pp)
    panel_xy(im, 3.65, -0.1, 5.7, 4.15, pp)
    draw_text(
        im, pxy(-3.45, 1.48), "normalized risk-weight simplex", 21, CYAN, pp, bold=True
    )
    draw_text(im, pxy(3.65, 1.48), "curvature test", 21, GREEN, pp, bold=True)
    tri = [(-5.75, -1.35), (-1.15, -1.35), (-3.45, 1.15)]
    d = ImageDraw.Draw(im, "RGBA")
    aa = phase(t, 1.0, 1.8)
    d.polygon(
        [pxy(*q) for q in tri], fill=rgba(BLUE, 0.08 * aa), outline=rgba(CYAN, aa)
    )
    for q, tex, col, off in zip(
        tri,
        [r"q_1=1", r"q_2=1", r"q_3=1"],
        [BLUE, YELLOW, PURPLE],
        [(-0.05, -0.32), (0.05, -0.32), (0, -0.18)],
    ):
        paste_math(im, pxy(q[0] + off[0], q[1] + off[1]), tex, 20, col, aa)
    # Elliptical contours clipped approximately inside triangle.
    cont = phase(t, 1.55, 2.55)
    cx, cy = pxy(-3.45, -0.25)
    for k in range(4):
        ww = int(sc(3.2 - 0.46 * k))
        hh = int(sc(1.85 - 0.26 * k))
        d.ellipse(
            (cx - ww // 2, cy - hh // 2, cx + ww // 2, cy + hh // 2),
            outline=rgba(GREEN, 0.75 * cont),
            width=3,
        )
    start = np.array([-4.9, -1.02])
    end = np.array([-3.45, -0.25])
    u = phase(t, 2.35, 4.1)
    q = start * (1 - u) + end * u
    dot(im, (float(q[0]), float(q[1])), RED if u < 0.9 else GREEN, 0.105, 1)
    # Draw path.
    pts = [start, np.array([-4.35, -0.72]), np.array([-3.85, -0.43]), end]
    for a, b in zip(pts[:-1], pts[1:]):
        line(im, tuple(a), tuple(b), YELLOW, 3, phase(t, 2.3, 3.0))
    # Right matrix and formulas.
    vals = np.array(
        [
            [0, 0.4, 0.9, 0.6],
            [0.4, 0, 0.55, 0.8],
            [0.9, 0.55, 0, 0.3],
            [0.6, 0.8, 0.3, 0],
        ]
    )
    matrix_heatmap(im, 2.6, 0.35, vals, 1.65, phase(t, 1.5, 2.4), True)
    paste_math(
        im, pxy(4.75, 0.95), r"\mathbf{1}^{\top}u=0", 26, CYAN, phase(t, 2.2, 3.0)
    )
    paste_math(
        im,
        pxy(4.75, 0.35),
        r"u^{\top}\ell^{\circ2}u\leq 0",
        29,
        GREEN,
        phase(t, 2.65, 3.45),
    )
    paste_math(
        im,
        pxy(3.65, -0.72),
        r"C(q)\ \mathrm{concave}\quad\Longrightarrow\quad1-C(q)\ \mathrm{convex}",
        25,
        YELLOW,
        phase(t, 3.35, 4.35),
    )
    paste_math(
        im,
        pxy(3.65, -1.48),
        r"-\tfrac12H\ell^{\circ2}H\succeq0",
        29,
        FG,
        phase(t, 4.0, 4.9),
    )
    pill(im, (3.65, -2.08), "convex program", GREEN, phase(t, 4.55, 5.25), BG, 20)
    footer(
        im,
        "Thesis Theorem 5.5 — conditional negative definiteness is checkable directly from the observed floor matrix",
        phase(t, 5.0, 5.8),
    )
    return im


def scene13(t: float) -> Image.Image:
    im = base_frame()
    title(
        im,
        "13",
        "Three optimizations that look similar — but are not",
        "Track what is fixed and what is allowed to vary",
        phase(t, 0, 0.65),
    )
    panel_a = phase(t, 0.55, 1.25)
    centres = [-4.35, 0.0, 4.35]
    headings = [
        ("Essay I — pair", BLUE),
        ("Essay II — field", GREEN),
        ("Essay III — portfolio", PURPLE),
    ]
    for cx, (heading, colour) in zip(centres, headings):
        panel_xy(im, cx, -0.12, 4.0, 4.2, panel_a)
        draw_text(im, pxy(cx, 1.55), heading, 20, colour, panel_a, bold=True)

    # Pairwise coupling: fixed marginals, varying assignment.
    pair_a = phase(t, 1.0, 1.85)
    left_pts = [(-5.55, 0.65), (-5.55, 0.05), (-5.55, -0.55)]
    right_pts = [(-3.15, 0.65), (-3.15, 0.05), (-3.15, -0.55)]
    cloud(im, left_pts, BLUE, pair_a, 0.075)
    cloud(im, right_pts, YELLOW, pair_a, 0.075)
    perm0 = [2, 0, 1]
    perm1 = [0, 2, 1]
    move = phase(t, 2.05, 3.0)
    for k in range(3):
        y = lerp(right_pts[perm0[k]][1], right_pts[perm1[k]][1], move)
        line(im, left_pts[k], (right_pts[0][0], y), CYAN, 3, pair_a)
    pill(im, (-4.35, -1.05), "fixed: P_i, P_j", BLUE, phase(t, 2.65, 3.35), FG, 17)
    pill(im, (-4.35, -1.55), "varies: coupling pi", CYAN, phase(t, 2.85, 3.55), BG, 17)
    paste_math(
        im,
        pxy(-4.35, -2.05),
        r"\inf_{\pi\in\Pi(P_i,P_j)}",
        24,
        CYAN,
        phase(t, 3.05, 3.75),
    )

    # Target reconstruction: fixed target and assignments, varying simplex coordinates.
    field_a = phase(t, 1.35, 2.2)
    target_pts = [(-1.15, 0.65), (-0.85, 0.05), (-1.15, -0.55)]
    peer1 = [(0.05, 0.65), (0.35, 0.05), (0.05, -0.55)]
    peer2 = [(1.0, 0.65), (1.3, 0.05), (1.0, -0.55)]
    cloud(im, target_pts, BLUE, field_a, 0.075)
    cloud(im, peer1, YELLOW, field_a, 0.06)
    cloud(im, peer2, ORANGE, field_a, 0.06)
    links_a = phase(t, 1.9, 2.8)
    for m in range(3):
        dashed_line(im, target_pts[m], peer1[m], YELLOW, 2, links_a, 8, 6)
        dashed_line(im, target_pts[m], peer2[m], ORANGE, 2, links_a, 8, 6)
    # A small moving simplex split makes w_i visibly variable.
    w_u = 0.5 + 0.25 * math.sin(max(0.0, t - 2.5) * 2.1)
    bar_x1, bar_y = pxy(-0.85, -1.02)
    bar_x2, _ = pxy(1.15, -1.02)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle(
        (bar_x1, bar_y - 12, bar_x2, bar_y + 12), radius=10, fill=rgba(GRID, links_a)
    )
    split = int(lerp(bar_x1, bar_x2, clamp(w_u)))
    d.rounded_rectangle(
        (bar_x1, bar_y - 12, split, bar_y + 12), radius=10, fill=rgba(YELLOW, links_a)
    )
    d.rounded_rectangle(
        (split, bar_y - 12, bar_x2, bar_y + 12), radius=10, fill=rgba(ORANGE, links_a)
    )
    pill(
        im,
        (0, -1.48),
        "fixed: target + alignments",
        GREEN,
        phase(t, 2.65, 3.35),
        BG,
        16,
    )
    pill(
        im,
        (0, -1.88),
        "varies: simplex weights w_i",
        YELLOW,
        phase(t, 2.85, 3.55),
        BG,
        16,
    )
    paste_math(
        im, pxy(0, -2.28), r"\min_{w_i\in\Delta_{-i}}", 24, YELLOW, phase(t, 3.05, 3.75)
    )

    # Portfolio dispersion: fixed q, varying one coherent coupling / free centre.
    port_a = phase(t, 1.65, 2.5)
    laws = [(3.25, 0.6), (5.45, 0.6), (4.35, -0.55)]
    law_colours = [BLUE, YELLOW, PURPLE]
    for point, colour in zip(laws, law_colours):
        circle(im, point, 0.34, colour, port_a, 3, 0.06)
    qx = 4.35 + 0.18 * math.sin(max(0.0, t - 2.3) * 1.8)
    qy = 0.05 + 0.12 * math.cos(max(0.0, t - 2.3) * 1.8)
    for point, colour in zip(laws, law_colours):
        dashed_line(im, point, (qx, qy), colour, 2, phase(t, 2.15, 3.05), 8, 6)
    dot(im, (qx, qy), GREEN, 0.12, phase(t, 2.15, 2.85))
    paste_math(im, pxy(qx, qy + 0.3), r"Q", 25, GREEN, phase(t, 2.15, 2.85))
    pill(
        im,
        (4.35, -1.12),
        "fixed inside D_q: portfolio q",
        PURPLE,
        phase(t, 2.65, 3.35),
        FG,
        16,
    )
    pill(
        im,
        (4.35, -1.55),
        "varies: common gamma or Q",
        GREEN,
        phase(t, 2.85, 3.55),
        BG,
        16,
    )
    paste_math(
        im,
        pxy(4.35, -2.02),
        r"\inf_{\gamma}\;\equiv\;\inf_Q",
        24,
        GREEN,
        phase(t, 3.05, 3.75),
    )

    takeaway = phase(t, 4.0, 5.0)
    draw_text(
        im,
        pxy(0, -2.78),
        "Different fixed objects  →  different financial questions",
        24,
        FG,
        takeaway,
        bold=True,
    )
    draw_text(
        im,
        pxy(0, -3.15),
        "Only the outer portfolio problem varies q after fixed-q dispersion is evaluated",
        18,
        MUTED,
        takeaway,
    )
    footer(
        im,
        "Thesis §2.2 — separation, target reconstruction and dispersion are distinct optimization layers",
        phase(t, 5.0, 5.85),
    )
    return im


SCENES: list[tuple[str, str, Callable[[float], Image.Image]]] = [
    ("01", "firm_as_distribution", scene01),
    ("02", "couplings_and_wasserstein", scene02),
    ("03", "polarization_to_covariance", scene03),
    ("04", "covariance_envelope", scene04),
    ("05", "transmission_and_slack", scene05),
    ("06", "target_anchored_reconstruction", scene06),
    ("07", "symmetric_distance_directed_field", scene07),
    ("08", "quadratic_adjustment_spatial_closure", scene08),
    ("09", "pairwise_not_jointly_coherent", scene09),
    ("10", "multifirm_dispersion_barycentre", scene10),
    ("11", "portfolio_variance_certificate", scene11),
    ("12", "certificate_convexity", scene12),
    ("13", "three_optimizations", scene13),
]


def render_video(
    func: Callable[[float], Image.Image],
    out: Path,
    duration: float = DURATION,
    fps: int = FPS,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    total = int(round(duration * fps))
    try:
        for i in range(total):
            t = i / fps
            frame = func(t).convert("RGB")
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with code {rc}: {out}")


def concat_videos(paths: Sequence[Path], out: Path) -> None:
    listing = ROOT / "tools" / "concat.txt"
    listing.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in paths), encoding="utf-8"
    )
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
            str(out),
        ],
        check=True,
    )


def build_contact_sheet(
    items: Sequence[tuple[str, str, Callable[[float], Image.Image]]],
) -> None:
    columns = 4
    rows = math.ceil(len(items) / columns)
    thumb_w, thumb_h = 400, 225
    sheet = Image.new("RGB", (thumb_w * columns, thumb_h * rows), ImageColor.getrgb(BG))
    for idx, (num, name, func) in enumerate(items):
        frame = (
            func(6.2)
            .convert("RGB")
            .resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        )
        x = (idx % columns) * thumb_w
        y = (idx // columns) * thumb_h
        sheet.paste(frame, (x, y))
        draw = ImageDraw.Draw(sheet, "RGBA")
        draw.rectangle((x, y, x + thumb_w, y + 30), fill=rgba(BG, 0.74))
        draw.text(
            (x + 9, y + 15),
            f"{num}  {name.replace('_', ' ')}",
            font=font(14, bold=True),
            fill=rgba(FG),
            anchor="lm",
        )
    sheet.save(THUMB_DIR / "contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", action="append", help="scene number, e.g. 01; repeatable"
    )
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--thumbnails-only", action="store_true")
    args = parser.parse_args()
    VIDEO_DIR.mkdir(exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)
    chosen = [s for s in SCENES if not args.scene or s[0] in args.scene]
    outputs = []
    for num, name, func in chosen:
        thumb = func(6.2).convert("RGB")
        thumb.save(THUMB_DIR / f"{num}_{name}.png")
        if not args.thumbnails_only:
            out = VIDEO_DIR / f"{num}_{name}.mp4"
            print(f"Rendering {out.name}...")
            render_video(func, out, args.duration, args.fps)
            outputs.append(out)
    build_contact_sheet(SCENES)
    if not args.scene and not args.thumbnails_only:
        concat_videos(outputs, VIDEO_DIR / "00_full_series.mp4")
        print("Created 00_full_series.mp4")


if __name__ == "__main__":
    main()
