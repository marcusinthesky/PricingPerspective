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
    ORANGE,
    PURPLE,
    panel,
    label,
    formula,
    pill,
    title_block,
    footer,
)


class QuadraticAdjustmentSpatialClosure(Scene):
    """A quadratic exposure-adjustment problem yields the spatial autoregression."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "08",
            "Quadratic adjustment implies spatial closure",
            "Deriving the spatial lag at the exposure layer",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        p = panel(12.0, 4.2).move_to(DOWN * 0.1)
        self.play(Create(p))
        baseline = NumberLine(
            x_range=[-3, 3, 1],
            length=9.0,
            include_numbers=False,
            include_ticks=False,
            color=MUTED,
        )
        baseline.move_to(UP * 0.65)
        xi_pos = baseline.n2p(-2.0)
        peer_pos = baseline.n2p(2.0)
        b_pos = baseline.n2p(0.75)
        xi = Dot(xi_pos, radius=0.14, color=BLUE)
        peer = Dot(peer_pos, radius=0.14, color=YELLOW)
        b = Dot(b_pos, radius=0.16, color=GREEN)
        lxi = formula(r"\xi_i", 0.75, BLUE).next_to(xi, UP, buff=0.12)
        lp = formula(r"\sum_jW_{ij}B_j", 0.66, YELLOW).next_to(peer, UP, buff=0.12)
        lb = formula(r"B_i", 0.75, GREEN).next_to(b, DOWN, buff=0.12)
        self.play(Create(baseline), FadeIn(xi), FadeIn(peer), FadeIn(lxi), FadeIn(lp))
        self.play(GrowFromCenter(b), FadeIn(lb))

        left_brace = BraceBetweenPoints(xi_pos, b_pos, DOWN, color=BLUE)
        right_brace = BraceBetweenPoints(b_pos, peer_pos, DOWN, color=YELLOW)
        c1 = label("departure from stand-alone exposure", BLUE, 18).next_to(
            left_brace, DOWN, buff=0.06
        )
        c2 = label("peer misalignment", YELLOW, 18).next_to(
            right_brace, DOWN, buff=0.06
        )
        self.play(
            GrowFromCenter(left_brace),
            GrowFromCenter(right_brace),
            FadeIn(c1),
            FadeIn(c2),
        )

        objective = formula(
            r"\min_a\ \frac12\|a-\xi_i\|^2+\frac\lambda2\sum_jW_{ij}\|a-B_j\|^2", 0.73
        )
        objective.move_to(DOWN * 1.35)
        self.play(Write(objective))

        rho_map = (
            VGroup(
                pill("lambda = 0", BLUE, FG),
                Arrow(LEFT, RIGHT, color=MUTED, stroke_width=2).scale(0.45),
                pill("rho = lambda / (1 + lambda)", GREEN, BG),
                Arrow(LEFT, RIGHT, color=MUTED, stroke_width=2).scale(0.45),
                pill("0 <= rho < 1", YELLOW, BG),
            )
            .arrange(RIGHT, buff=0.18)
            .move_to(DOWN * 2.1)
        )
        self.play(FadeIn(rho_map, shift=UP * 0.12))

        closure = formula(r"B=\rho WB+(1-\rho)\xi", 0.98, GREEN).to_edge(
            DOWN, buff=0.48
        )
        self.play(ReplacementTransform(objective, closure), run_time=1.2)
        self.play(
            FadeIn(
                footer(
                    "Thesis Theorems 4.1–4.2 — the multiplier (1−ρ)(I−ρW)⁻¹ accumulates direct and higher-order peer adjustment"
                )
            )
        )
        self.wait(1.2)
