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
    pill,
    title_block,
    footer,
)


class ThreeOptimizations(Scene):
    """Contrast the fixed and varying objects in the three optimization layers."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "13",
            "Three optimizations that look similar — but are not",
            "Track what is fixed and what is allowed to vary",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        cards = VGroup(*[panel(4.15, 4.25) for _ in range(3)]).arrange(RIGHT, buff=0.25)
        cards.move_to(DOWN * 0.12)
        self.play(LaggedStart(*[Create(card) for card in cards], lag_ratio=0.15))

        headings = [
            ("Essay I — pair", BLUE),
            ("Essay II — field", GREEN),
            ("Essay III — portfolio", PURPLE),
        ]
        for card, (text, color) in zip(cards, headings):
            self.play(
                FadeIn(label(text, color, 21).next_to(card.get_top(), DOWN, buff=0.16)),
                run_time=0.25,
            )

        # Essay I: fixed marginal laws, varying pairwise coupling.
        c1 = cloud([(-5.75, 0.55), (-5.75, -0.05), (-5.75, -0.65)], BLUE, 0.075)
        c2 = cloud([(-3.35, 0.55), (-3.35, -0.05), (-3.35, -0.65)], YELLOW, 0.075)
        pair_lines = VGroup(
            Line(c1[0], c2[2], color=CYAN, stroke_width=2.5),
            Line(c1[1], c2[0], color=CYAN, stroke_width=2.5),
            Line(c1[2], c2[1], color=CYAN, stroke_width=2.5),
        )
        self.play(
            FadeIn(c1),
            FadeIn(c2),
            LaggedStart(*[Create(line) for line in pair_lines], lag_ratio=0.12),
        )
        pair_tags = (
            VGroup(
                pill("fixed: P_i, P_j", BLUE, FG),
                pill("varies: coupling pi", CYAN, BG),
            )
            .arrange(DOWN, buff=0.12)
            .move_to(cards[0].get_center() + DOWN * 1.25)
        )
        self.play(FadeIn(pair_tags))

        # Essay II: fixed target and correspondences, varying simplex coordinates.
        target = cloud([(-0.7, 0.65), (-0.35, 0.0), (-0.65, -0.65)], BLUE, 0.08)
        peer1 = cloud([(0.25, 0.65), (0.55, 0.0), (0.25, -0.65)], YELLOW, 0.065)
        peer2 = cloud([(1.05, 0.65), (1.35, 0.0), (1.05, -0.65)], ORANGE, 0.065)
        links = VGroup(
            *[
                DashedLine(
                    target[m],
                    peer1[m],
                    color=YELLOW,
                    stroke_width=1.2,
                    stroke_opacity=0.55,
                )
                for m in range(3)
            ],
            *[
                DashedLine(
                    target[m],
                    peer2[m],
                    color=ORANGE,
                    stroke_width=1.2,
                    stroke_opacity=0.55,
                )
                for m in range(3)
            ],
        )
        self.play(
            FadeIn(target),
            FadeIn(peer1),
            FadeIn(peer2),
            LaggedStart(*[Create(link) for link in links], lag_ratio=0.08),
        )
        field_tags = (
            VGroup(
                pill("fixed: target + alignments", GREEN, BG),
                pill("varies: simplex weights w_i", YELLOW, BG),
            )
            .arrange(DOWN, buff=0.12)
            .move_to(cards[1].get_center() + DOWN * 1.25)
        )
        self.play(FadeIn(field_tags))

        # Essay III: fixed q inside dispersion, varying one coherent law or centre.
        law_centres = [
            np.array([3.0, 0.55, 0]),
            np.array([5.55, 0.55, 0]),
            np.array([4.3, -0.55, 0]),
        ]
        law_colors = [BLUE, YELLOW, PURPLE]
        laws = VGroup(
            *[
                Circle(radius=0.38, color=color, stroke_width=2.5)
                .set_fill(color, opacity=0.08)
                .move_to(point)
                for point, color in zip(law_centres, law_colors)
            ]
        )
        centre = Dot([4.3, 0.15, 0], radius=0.13, color=GREEN)
        centre_label = formula(r"Q", 0.65, GREEN).next_to(centre, UP, buff=0.08)
        spokes = VGroup(
            *[
                DashedLine(
                    point,
                    centre.get_center(),
                    color=color,
                    dash_length=0.1,
                    stroke_width=1.8,
                )
                for point, color in zip(law_centres, law_colors)
            ]
        )
        self.play(
            FadeIn(laws),
            GrowFromCenter(centre),
            FadeIn(centre_label),
            LaggedStart(*[Create(s) for s in spokes], lag_ratio=0.1),
        )
        portfolio_tags = (
            VGroup(
                pill("fixed inside D_q: portfolio q", PURPLE, FG),
                pill("varies: common gamma or Q", GREEN, BG),
            )
            .arrange(DOWN, buff=0.12)
            .move_to(cards[2].get_center() + DOWN * 1.25)
        )
        self.play(FadeIn(portfolio_tags))

        formulas = VGroup(
            formula(r"\inf_{\pi\in\Pi(P_i,P_j)}", 0.58, CYAN),
            formula(r"\min_{w_i\in\Delta_{-i}}", 0.58, YELLOW),
            formula(r"\inf_{\gamma}\;\equiv\;\inf_Q", 0.58, GREEN),
        )
        for expression, card in zip(formulas, cards):
            expression.next_to(card.get_bottom(), UP, buff=0.18)
        self.play(
            LaggedStart(*[Write(expression) for expression in formulas], lag_ratio=0.15)
        )

        outer = (
            VGroup(
                label("Only after evaluating dispersion at fixed q:", MUTED, 19),
                formula(r"\text{outer portfolio problem varies }q", 0.62, FG),
            )
            .arrange(RIGHT, buff=0.18)
            .to_edge(DOWN, buff=0.46)
        )
        self.play(FadeIn(outer, shift=UP * 0.1))
        self.play(
            FadeIn(
                footer(
                    "Thesis §2.2 — separation, target reconstruction, and dispersion solve different optimization problems"
                )
            )
        )
        self.wait(1.2)
