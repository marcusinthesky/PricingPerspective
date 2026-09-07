from manim import *
import numpy as np
from theme import (
    BG,
    FG,
    MUTED,
    BLUE,
    CYAN,
    YELLOW,
    GREEN,
    RED,
    PURPLE,
    panel,
    label,
    formula,
    title_block,
    footer,
)


class PolarizationToCovariance(Scene):
    """With magnitudes fixed, squared distance and inner product move in opposite directions."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "03",
            "Polarization: distance becomes covariance",
            "The key algebraic bridge from transport geometry to second moments",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        left = panel(6.0, 4.1).move_to(LEFT * 3.25 + DOWN * 0.15)
        right = panel(5.4, 4.1).move_to(RIGHT * 3.55 + DOWN * 0.15)
        self.play(Create(left), Create(right))

        origin = left.get_center() + DOWN * 0.45
        axes = VGroup(
            Line(
                origin + LEFT * 2.25, origin + RIGHT * 2.25, color=MUTED, stroke_width=2
            ),
            Line(origin + DOWN * 1.45, origin + UP * 1.45, color=MUTED, stroke_width=2),
        )
        zi_radius, zj_radius = 1.80, 1.65
        zi_angle = 8 * DEGREES
        zj_angle = 70 * DEGREES
        zi_end = origin + zi_radius * np.array(
            [np.cos(zi_angle), np.sin(zi_angle), 0.0]
        )
        zj_end = origin + zj_radius * np.array(
            [np.cos(zj_angle), np.sin(zj_angle), 0.0]
        )
        zi = Arrow(origin, zi_end, buff=0, color=BLUE, stroke_width=5)
        zj = Arrow(origin, zj_end, buff=0, color=YELLOW, stroke_width=5)
        chord = DashedLine(zi_end, zj_end, color=RED, dash_length=0.12)
        li = formula(r"z_i", 0.75, BLUE).next_to(zi_end, RIGHT, buff=0.08)
        lj = formula(r"z_j", 0.75, YELLOW).next_to(zj_end, UP, buff=0.08)
        self.play(
            Create(axes),
            GrowArrow(zi),
            GrowArrow(zj),
            FadeIn(li),
            FadeIn(lj),
            Create(chord),
        )

        identity = formula(
            r"\langle z_i,z_j\rangle=\tfrac12\bigl(\|z_i\|^2+\|z_j\|^2-\|z_i-z_j\|^2\bigr)",
            0.66,
        )
        identity.move_to(right.get_center() + UP * 0.9)
        self.play(Write(identity))

        fixed = label("Fix the two marginal magnitudes", MUTED, 21).move_to(
            right.get_center() + UP * 0.05
        )
        relation = (
            VGroup(
                label("smaller squared distance", GREEN, 22),
                formula(r"\Longleftrightarrow", 0.8, FG),
                label("larger inner product", PURPLE, 22),
            )
            .arrange(DOWN, buff=0.17)
            .move_to(right.get_center() + DOWN * 0.65)
        )
        self.play(
            FadeIn(fixed),
            LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in relation], lag_ratio=0.15),
        )

        zj_close_angle = 20 * DEGREES
        zj_close_end = origin + zj_radius * np.array(
            [np.cos(zj_close_angle), np.sin(zj_close_angle), 0.0]
        )
        zj_close = Arrow(origin, zj_close_end, buff=0, color=YELLOW, stroke_width=5)
        chord_close = DashedLine(zi_end, zj_close_end, color=GREEN, dash_length=0.12)
        self.play(
            Transform(zj, zj_close),
            Transform(chord, chord_close),
            lj.animate.next_to(zj_close_end, UP, buff=0.08),
            run_time=1.4,
        )

        consequence = formula(
            r"\min_{\pi}\mathbb E_\pi\|Z_i-Z_j\|^2\quad\Longleftrightarrow\quad\max_{\pi}\mathbb E_\pi\langle Z_i,Z_j\rangle",
            0.72,
        )
        consequence.to_edge(DOWN, buff=0.48)
        self.play(Write(consequence))
        self.play(
            FadeIn(
                footer(
                    "Thesis Eq. (1.3), Lemma 3.1 — W₂ has a special covariance role because its cost is squared distance"
                )
            )
        )
        self.wait(1.2)
