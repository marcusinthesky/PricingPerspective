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
    ORANGE,
    PURPLE,
    panel,
    label,
    formula,
    title_block,
    footer,
)


class PortfolioVarianceCertificate(Scene):
    """Weighted polarization writes risk as perfect alignment minus certified dispersion."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "11",
            "Portfolio variance as alignment minus dispersion",
            "Information rules out part of the worst-case risk benchmark",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        p = panel(12.0, 4.15).move_to(DOWN * 0.1)
        self.play(Create(p))
        origin = np.array([-4.5, -0.7, 0])
        vecs_aligned = VGroup(
            *[
                Arrow(
                    origin,
                    origin + np.array([2.4 + 0.25 * k, 1.3 + 0.12 * k, 0]),
                    buff=0,
                    color=col,
                    stroke_width=5,
                    tip_length=0.18,
                )
                for k, col in enumerate([BLUE, YELLOW, PURPLE, ORANGE])
            ]
        )
        aligned_label = label("perfect positive alignment", RED, 22).move_to(
            [-3.2, 1.55, 0]
        )
        self.play(
            FadeIn(aligned_label),
            LaggedStart(*[GrowArrow(v) for v in vecs_aligned], lag_ratio=0.12),
        )

        bar_x = 1.0
        bar_bg = (
            RoundedRectangle(corner_radius=0.12, width=1.4, height=3.0)
            .move_to([bar_x, 0.1, 0])
            .set_fill(RED, opacity=0.22)
            .set_stroke(RED, width=2)
        )
        bar_full = (
            Rectangle(width=1.0, height=2.55)
            .move_to([bar_x, 0.02, 0])
            .set_fill(RED, opacity=0.8)
            .set_stroke(width=0)
        )
        b_label = label("risk cap", FG, 20).next_to(bar_bg, UP, buff=0.12)
        one = formula(r"\sum_iq_iv_i", 0.58, RED).move_to(bar_full)
        self.play(Create(bar_bg), FadeIn(bar_full), FadeIn(b_label), FadeIn(one))

        dispersed = VGroup(
            Arrow(
                origin,
                origin + np.array([2.6, 1.2, 0]),
                buff=0,
                color=BLUE,
                stroke_width=5,
            ),
            Arrow(
                origin,
                origin + np.array([1.7, 2.0, 0]),
                buff=0,
                color=YELLOW,
                stroke_width=5,
            ),
            Arrow(
                origin,
                origin + np.array([2.0, -0.2, 0]),
                buff=0,
                color=PURPLE,
                stroke_width=5,
            ),
            Arrow(
                origin,
                origin + np.array([0.8, 1.7, 0]),
                buff=0,
                color=ORANGE,
                stroke_width=5,
            ),
        )
        self.play(
            Transform(vecs_aligned, dispersed),
            Transform(
                aligned_label,
                label("certified separation floors", GREEN, 22).move_to(
                    [-3.2, 1.55, 0]
                ),
            ),
            run_time=1.5,
        )

        cut = (
            Rectangle(width=1.0, height=0.8)
            .align_to(bar_full, DOWN)
            .move_to([bar_x, -0.85, 0])
            .set_fill(GREEN, opacity=0.95)
            .set_stroke(width=0)
        )
        cut_label = formula(r"C(q)", 0.62, BG).move_to(cut)
        new_bar = (
            Rectangle(width=1.0, height=1.75)
            .align_to(bar_full, UP)
            .move_to([bar_x, 0.42, 0])
            .set_fill(YELLOW, opacity=0.85)
            .set_stroke(width=0)
        )
        self.play(
            FadeIn(cut, shift=UP * 0.1),
            FadeIn(cut_label),
            Transform(bar_full, new_bar),
            FadeOut(one),
        )

        right_text = (
            VGroup(
                label("not a covariance estimate", MUTED, 21),
                label("a one-sided certificate", GREEN, 24),
                label("about admissible portfolio risk", MUTED, 21),
            )
            .arrange(DOWN, buff=0.12)
            .move_to([4.1, 0.2, 0])
        )
        self.play(
            LaggedStart(
                *[FadeIn(t, shift=UP * 0.1) for t in right_text], lag_ratio=0.15
            )
        )

        eq = formula(
            r"V_{\rm sys}(q)\le\sum_iq_iv_i-\frac12\sum_{i,j}q_iq_j\ell_{ij}^2", 0.83
        )
        eq.to_edge(DOWN, buff=0.46)
        self.play(Write(eq))
        self.play(
            FadeIn(
                footer(
                    "Thesis Theorem 5.1 — observed separation certifies diversification relative to perfect alignment"
                )
            )
        )
        self.wait(1.2)
