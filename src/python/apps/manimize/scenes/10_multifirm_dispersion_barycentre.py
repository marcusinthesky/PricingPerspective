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
    cloud,
    label,
    formula,
    title_block,
    footer,
)


class MultifirmDispersionBarycentre(Scene):
    """For fixed portfolio weights, coherent dispersion varies a common coupling or a free centre law."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "10",
            "Multi-firm transport dispersion",
            "One coherent coupling — equivalently, a free Wasserstein centre",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        left = panel(6.2, 4.2).move_to(LEFT * 3.45 + DOWN * 0.1)
        right = panel(5.7, 4.2).move_to(RIGHT * 3.65 + DOWN * 0.1)
        self.play(Create(left), Create(right))
        self.play(
            FadeIn(
                label("common-coupling view", CYAN, 22).next_to(
                    left.get_top(), DOWN, buff=0.18
                )
            ),
            FadeIn(
                label("free-centre view", GREEN, 22).next_to(
                    right.get_top(), DOWN, buff=0.18
                )
            ),
        )

        centers = [
            np.array([-5.1, 0.55, 0]),
            np.array([-3.35, 0.55, 0]),
            np.array([-1.6, 0.55, 0]),
        ]
        colors = [BLUE, YELLOW, PURPLE]
        clouds = VGroup()
        for c, col in zip(centers, colors):
            pts = [
                (c[0] - 0.25, c[1] + 0.55),
                (c[0] + 0.2, c[1] + 0.2),
                (c[0] - 0.1, c[1] - 0.25),
                (c[0] + 0.3, c[1] - 0.65),
            ]
            clouds.add(cloud(pts, col, 0.085))
        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in clouds], lag_ratio=0.16)
        )
        names = VGroup(
            *[
                formula(rf"C_{i + 1}", 0.62, col).next_to(clouds[i], DOWN, buff=0.12)
                for i, col in enumerate(colors)
            ]
        )
        self.play(FadeIn(names))

        triples = VGroup()
        for m in range(4):
            triangle = VGroup(
                Line(clouds[0][m], clouds[1][m], color=MUTED, stroke_width=1.4),
                Line(clouds[1][m], clouds[2][m], color=MUTED, stroke_width=1.4),
                Line(clouds[0][m], clouds[2][m], color=MUTED, stroke_width=1.4),
            )
            triples.add(triangle)
        self.play(LaggedStart(*[Create(t) for t in triples], lag_ratio=0.15))
        common = label("each realization joins all firms at once", MUTED, 18).next_to(
            left.get_bottom(), UP, buff=0.18
        )
        self.play(FadeIn(common))

        right_centers = [
            np.array([1.55, 0.75, 0]),
            np.array([4.65, 0.75, 0]),
            np.array([3.1, -1.0, 0]),
        ]
        rc = VGroup(
            *[
                Circle(radius=0.58, color=col, stroke_width=3).move_to(c)
                for c, col in zip(right_centers, colors)
            ]
        )
        rn = VGroup(
            *[
                formula(rf"P_{i + 1}", 0.62, col).move_to(c)
                for i, (c, col) in enumerate(zip(right_centers, colors))
            ]
        )
        self.play(LaggedStart(*[Create(c) for c in rc], lag_ratio=0.16), FadeIn(rn))
        q_center = Dot([3.1, 0.15, 0], radius=0.16, color=GREEN)
        q_label = formula(r"Q", 0.8, GREEN).next_to(q_center, UP, buff=0.1)
        q_lines = VGroup(
            *[
                DashedLine(c, q_center.get_center(), color=col, dash_length=0.1)
                for c, col in zip(right_centers, colors)
            ]
        )
        self.play(
            GrowFromCenter(q_center),
            FadeIn(q_label),
            LaggedStart(*[Create(l) for l in q_lines], lag_ratio=0.12),
        )

        eq1 = formula(
            r"D_q(C_1,\ldots,C_n)=\inf_{\gamma}\mathbb E_\gamma\sum_{i<j}q_iq_jd(X_i,X_j)^2",
            0.58,
        )
        eq2 = formula(r"D_q(P_1,\ldots,P_n)=\inf_Q\sum_iq_iW_2^2(P_i,Q)", 0.7, GREEN)
        self.play(Write(eq1), Write(eq2))
        self.play(
            FadeIn(
                footer(
                    "Thesis Eqs. (5.9)–(5.11) — q is fixed inside Dq; the outer portfolio problem varies q later"
                )
            )
        )
        self.wait(1.2)
