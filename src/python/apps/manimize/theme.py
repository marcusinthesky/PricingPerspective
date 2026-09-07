"""Shared styling and helpers for the manuscript animation series.

Target runtime: Manim Community Edition 0.21.x.
"""

from __future__ import annotations

import numpy as np
from manim import *
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

config.background_color = BG


def title_block(number: str, title: str, subtitle: str | None = None) -> VGroup:
    badge = RoundedRectangle(corner_radius=0.12, width=0.72, height=0.46)
    badge.set_fill(BLUE, opacity=1).set_stroke(BLUE, width=0)
    n = Text(number, font="DejaVu Sans", weight=BOLD, font_size=24, color=BG)
    n.move_to(badge)
    heading = Text(title, font="DejaVu Sans", weight=BOLD, font_size=34, color=FG)
    top = VGroup(badge, n, heading).arrange(RIGHT, buff=0.22)
    if subtitle:
        sub = Text(subtitle, font="DejaVu Sans", font_size=21, color=MUTED)
        block = VGroup(top, sub).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
    else:
        block = top
    block.to_edge(UP, buff=0.28).to_edge(LEFT, buff=0.38)
    return block


def footer(text: str) -> Text:
    obj = Text(text, font="DejaVu Sans", font_size=17, color=MUTED)
    obj.to_edge(DOWN, buff=0.18)
    return obj


def panel(width: float, height: float, color: str = PANEL) -> RoundedRectangle:
    p = RoundedRectangle(corner_radius=0.18, width=width, height=height)
    return p.set_fill(color, opacity=0.78).set_stroke(GRID, width=1.4)


def cloud(points, color=BLUE, radius: float = 0.075, opacity: float = 1.0) -> VGroup:
    return VGroup(
        *[
            Dot(
                point=np.array([x, y, 0.0]),
                radius=radius,
                color=color,
                fill_opacity=opacity,
            )
            for x, y in points
        ]
    )


def centroid_marker(point, color=FG) -> VGroup:
    x, y = float(point[0]), float(point[1])
    a = Line([x - 0.12, y, 0], [x + 0.12, y, 0], color=color, stroke_width=2)
    b = Line([x, y - 0.12, 0], [x, y + 0.12, 0], color=color, stroke_width=2)
    return VGroup(a, b)


def label(text: str, color=FG, size=22) -> Text:
    return Text(text, font="DejaVu Sans", font_size=size, color=color)


def pill(text: str, color=BLUE, text_color=BG) -> VGroup:
    t = label(text, text_color, 18)
    r = RoundedRectangle(
        corner_radius=0.16, width=t.width + 0.34, height=t.height + 0.18
    )
    r.set_fill(color, opacity=1).set_stroke(color, width=0)
    t.move_to(r)
    return VGroup(r, t)


def curved_arrow(start, end, color=BLUE, angle=0.25, stroke_width=3) -> CurvedArrow:
    return CurvedArrow(
        start, end, angle=angle, color=color, stroke_width=stroke_width, tip_length=0.16
    )


def formula(tex: str, scale: float = 0.78, color=FG) -> MathTex:
    return MathTex(tex, color=color).scale(scale)


def section_line(y: float = 2.35) -> Line:
    return Line(LEFT * 6.65 + UP * y, RIGHT * 6.65 + UP * y, color=GRID, stroke_width=1)
