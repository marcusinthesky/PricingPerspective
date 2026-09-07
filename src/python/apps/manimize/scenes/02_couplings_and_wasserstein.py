from manim import *
from theme import (
    BG,
    FG,
    MUTED,
    BLUE,
    CYAN,
    YELLOW,
    GREEN,
    RED,
    panel,
    cloud,
    label,
    formula,
    title_block,
    footer,
)


class CouplingsAndWasserstein(Scene):
    """A coupling is a joint assignment; W2 selects the least squared displacement."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "02",
            "Couplings and quadratic transport",
            "Which observations from two distributions occur together?",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        p = panel(12.1, 4.0).move_to(DOWN * 0.15)
        self.play(Create(p))
        left_x, right_x = -4.2, 4.2
        ys = [1.05, 0.35, -0.35, -1.05]
        left_pts = [(left_x, y) for y in ys]
        right_pts = [(right_x, y) for y in [0.9, -0.9, 0.2, -0.2]]
        A = cloud(left_pts, BLUE, radius=0.105)
        B = cloud(right_pts, YELLOW, radius=0.105)
        la = formula(r"C_i", 0.72, BLUE).next_to(A, LEFT, buff=0.35)
        lb = formula(r"C_j", 0.72, YELLOW).next_to(B, RIGHT, buff=0.35)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in A], lag_ratio=0.08),
            LaggedStart(*[GrowFromCenter(d) for d in B], lag_ratio=0.08),
            FadeIn(la),
            FadeIn(lb),
        )

        bad_perm = [1, 0, 3, 2]
        good_perm = [0, 2, 3, 1]
        bad_lines = VGroup(
            *[
                Line(
                    A[k].get_center(),
                    B[bad_perm[k]].get_center(),
                    color=RED,
                    stroke_width=3,
                    stroke_opacity=0.8,
                )
                for k in range(4)
            ]
        )
        bad_tag = label("one feasible coupling: large displacement", RED, 22).move_to(
            UP * 1.75
        )
        self.play(
            LaggedStart(*[Create(line) for line in bad_lines], lag_ratio=0.08),
            FadeIn(bad_tag),
        )
        self.wait(0.5)

        good_lines = VGroup(
            *[
                Line(
                    A[k].get_center(),
                    B[good_perm[k]].get_center(),
                    color=GREEN,
                    stroke_width=3,
                    stroke_opacity=0.9,
                )
                for k in range(4)
            ]
        )
        good_tag = label(
            "optimal coupling: minimum average squared displacement", GREEN, 22
        ).move_to(UP * 1.75)
        self.play(
            ReplacementTransform(bad_lines, good_lines),
            Transform(bad_tag, good_tag),
            run_time=1.5,
        )

        brace = Brace(good_lines, DOWN, color=CYAN)
        cost = formula(
            r"W_2^2(C_i,C_j)=\inf_{\pi\in\Pi(C_i,C_j)}\;\mathbb E_\pi[d(X,Y)^2]", 0.78
        )
        cost.to_edge(DOWN, buff=0.45)
        self.play(GrowFromCenter(brace), Write(cost))
        self.play(
            FadeIn(
                footer(
                    "Thesis Eq. (1.2) — transport uses the geometry of the underlying state space"
                )
            )
        )
        self.wait(1.2)
