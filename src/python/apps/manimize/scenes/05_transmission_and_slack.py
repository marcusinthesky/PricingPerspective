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


class TransmissionAndSlack(Scene):
    """Observable information distance becomes a conditional risk-space bracket."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "05",
            "Transmission, distortion and slack",
            "Why embedding distance is not silently equated with financial risk distance",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        info_panel = panel(5.0, 3.7).move_to(LEFT * 3.8 + DOWN * 0.15)
        risk_panel = panel(5.0, 3.7).move_to(RIGHT * 3.8 + DOWN * 0.15)
        info_name = label("observable information space", BLUE, 22).next_to(
            info_panel.get_top(), DOWN, buff=0.2
        )
        risk_name = label("latent factor-risk space", ORANGE, 22).next_to(
            risk_panel.get_top(), DOWN, buff=0.2
        )
        self.play(
            Create(info_panel), Create(risk_panel), FadeIn(info_name), FadeIn(risk_name)
        )

        c1 = cloud(
            [(-5.0, 0.55), (-4.75, 0.25), (-4.9, -0.15), (-4.55, -0.5)], BLUE, 0.09
        )
        c2 = cloud(
            [(-2.95, 0.55), (-2.7, 0.2), (-2.85, -0.2), (-2.5, -0.55)], CYAN, 0.09
        )
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in c1], lag_ratio=0.06),
            LaggedStart(*[GrowFromCenter(d) for d in c2], lag_ratio=0.06),
        )
        g_line = DoubleArrow(
            c1.get_center(),
            c2.get_center(),
            color=GREEN,
            stroke_width=3,
            tip_length=0.15,
        )
        g = formula(r"G_{ij}=W_2(C_i,C_j)", 0.65, GREEN).next_to(g_line, UP, buff=0.08)
        self.play(GrowArrow(g_line), FadeIn(g))

        arrow = Arrow(
            info_panel.get_right() + RIGHT * 0.15,
            risk_panel.get_left() + LEFT * 0.15,
            color=YELLOW,
            stroke_width=5,
            tip_length=0.25,
        )
        tlabel = (
            VGroup(
                formula(r"T(x,U)", 0.75, YELLOW),
                label("common randomized carrier", MUTED, 18),
            )
            .arrange(DOWN, buff=0.05)
            .next_to(arrow, UP, buff=0.12)
        )
        self.play(GrowArrow(arrow), FadeIn(tlabel))

        p1_center = np.array([2.8, 0.0, 0])
        p2_center = np.array([4.8, 0.0, 0])
        p1 = cloud(
            [(2.45, 0.45), (2.85, 0.2), (2.55, -0.3), (3.05, -0.55)], ORANGE, 0.09
        )
        p2 = cloud([(4.35, 0.45), (4.85, 0.25), (4.55, -0.25), (5.05, -0.5)], RED, 0.09)
        halo1 = Circle(
            radius=0.42, color=YELLOW, stroke_width=2, stroke_opacity=0.7
        ).move_to(p1_center)
        halo2 = Circle(
            radius=0.42, color=YELLOW, stroke_width=2, stroke_opacity=0.7
        ).move_to(p2_center)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in p1], lag_ratio=0.06),
            LaggedStart(*[GrowFromCenter(d) for d in p2], lag_ratio=0.06),
            Create(halo1),
            Create(halo2),
        )
        slack = formula(r"\tau_i", 0.6, YELLOW).next_to(halo1, DOWN, buff=0.03)
        slack2 = formula(r"\tau_j", 0.6, YELLOW).next_to(halo2, DOWN, buff=0.03)
        self.play(FadeIn(slack), FadeIn(slack2))

        wline = DoubleArrow(
            p1.get_center(),
            p2.get_center(),
            color=PURPLE,
            stroke_width=3,
            tip_length=0.15,
        )
        w = formula(r"W_{2,\Gamma}(P_i,P_j)", 0.62, CYAN).next_to(wline, UP, buff=0.08)
        self.play(GrowArrow(wline), FadeIn(w))

        bracket = formula(
            r"\bigl(L^{-1}G_{ij}-\tau_i-\tau_j\bigr)_+\le W_{2,\Gamma}(P_i,P_j)\le LG_{ij}+\tau_i+\tau_j",
            0.66,
        )
        bracket.to_edge(DOWN, buff=0.5)
        self.play(Write(bracket))
        self.play(
            FadeIn(
                footer(
                    "Thesis Corollary 3.1 — L and τ are maintained sensitivity inputs, not parameters estimated from returns"
                )
            )
        )
        self.wait(1.2)
