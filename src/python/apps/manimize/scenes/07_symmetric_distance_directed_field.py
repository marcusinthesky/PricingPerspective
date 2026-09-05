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


class SymmetricDistanceDirectedField(Scene):
    """Direction comes from target-specific reconstruction relevance, not asymmetric distance."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "07",
            "A symmetric metric can yield a directed field",
            "Distance measures separation; reconstruction measures conditional usefulness",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        left = panel(5.0, 4.0).move_to(LEFT * 3.8 + DOWN * 0.1)
        right = panel(6.3, 4.0).move_to(RIGHT * 3.15 + DOWN * 0.1)
        self.play(Create(left), Create(right))
        self.play(
            FadeIn(
                label("symmetric pairwise geometry", CYAN, 22).next_to(
                    left.get_top(), DOWN, buff=0.2
                )
            ),
            FadeIn(
                label("target-specific reconstruction", GREEN, 22).next_to(
                    right.get_top(), DOWN, buff=0.2
                )
            ),
        )

        A = Dot([-4.7, 0.55, 0], radius=0.16, color=BLUE)
        B = Dot([-2.9, 0.55, 0], radius=0.16, color=YELLOW)
        C = Dot([-3.8, -0.9, 0], radius=0.16, color=PURPLE)
        nodes = VGroup(A, B, C)
        labels = VGroup(
            label("A", BLUE, 22).next_to(A, UP, buff=0.1),
            label("B", YELLOW, 22).next_to(B, UP, buff=0.1),
            label("C", PURPLE, 22).next_to(C, DOWN, buff=0.1),
        )
        edges = VGroup(
            Line(A, B, color=MUTED), Line(B, C, color=MUTED), Line(C, A, color=MUTED)
        )
        self.play(FadeIn(nodes), FadeIn(labels), Create(edges))
        sym = formula(r"W_2(C_i,C_j)=W_2(C_j,C_i)", 0.68, CYAN).next_to(
            left.get_bottom(), UP, buff=0.25
        )
        self.play(Write(sym))

        names = ["A", "B", "C"]
        colors = [BLUE, YELLOW, PURPLE]
        coords = [
            np.array([1.5, 0.65, 0]),
            np.array([4.8, 0.8, 0]),
            np.array([3.15, -1.0, 0]),
        ]
        rn = VGroup(
            *[
                Dot(point, radius=0.18, color=color)
                for point, color in zip(coords, colors)
            ]
        )
        rl = VGroup(
            *[
                label(name, color, 23).next_to(
                    point, UP if name != "C" else DOWN, buff=0.1
                )
                for name, point, color in zip(names, coords, colors)
            ]
        )
        self.play(FadeIn(rn), FadeIn(rl))

        arrows = VGroup(
            CurvedArrow(
                coords[0] + RIGHT * 0.2,
                coords[1] + LEFT * 0.2,
                angle=-0.22,
                color=YELLOW,
                stroke_width=4,
                tip_length=0.16,
            ),
            CurvedArrow(
                coords[1] + LEFT * 0.15,
                coords[0] + RIGHT * 0.15,
                angle=-0.22,
                color=BLUE,
                stroke_width=2,
                tip_length=0.14,
            ),
            CurvedArrow(
                coords[0] + DOWN * 0.1,
                coords[2] + LEFT * 0.1,
                angle=0.2,
                color=PURPLE,
                stroke_width=3,
                tip_length=0.15,
            ),
            CurvedArrow(
                coords[2] + RIGHT * 0.1,
                coords[1] + DOWN * 0.1,
                angle=0.2,
                color=YELLOW,
                stroke_width=4,
                tip_length=0.16,
            ),
        )
        weights = VGroup(
            formula(r"0.60", 0.5, YELLOW).move_to([3.15, 1.15, 0]),
            formula(r"0.10", 0.5, BLUE).move_to([3.15, 0.3, 0]),
            formula(r"0.40", 0.5, PURPLE).move_to([2.15, -0.25, 0]),
            formula(r"0.75", 0.5, YELLOW).move_to([4.1, -0.25, 0]),
        )
        self.play(
            LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.15), FadeIn(weights)
        )
        asym = formula(r"W^\flat_{ij}\ne W^\flat_{ji}", 0.82, GREEN).next_to(
            right.get_bottom(), UP, buff=0.26
        )
        self.play(Write(asym))
        note = label(
            "direction = reconstruction relevance, not causal influence", MUTED, 20
        ).to_edge(DOWN, buff=0.48)
        self.play(FadeIn(note))
        self.play(
            FadeIn(
                footer(
                    "Thesis §4.4 — rows are nonnegative, sum to one, and assign zero self-weight"
                )
            )
        )
        self.wait(1.2)
