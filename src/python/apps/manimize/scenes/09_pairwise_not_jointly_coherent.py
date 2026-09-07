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
    cloud,
    label,
    formula,
    title_block,
    footer,
)


class PairwiseNotJointlyCoherent(Scene):
    """Three independently optimal balanced assignments can violate composition consistency."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "09",
            "Pairwise plans need not form one joint law",
            "Portfolio aggregation adds a compatibility requirement",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        cards = (
            VGroup(*[panel(3.9, 3.5) for _ in range(3)])
            .arrange(RIGHT, buff=0.28)
            .move_to(DOWN * 0.05)
        )
        self.play(LaggedStart(*[Create(c) for c in cards], lag_ratio=0.15))
        heads = ["A ↔ B optimum", "B ↔ C optimum", "A ↔ C optimum"]
        cols = [(BLUE, YELLOW), (YELLOW, PURPLE), (BLUE, PURPLE)]
        for card, head in zip(cards, heads):
            self.play(
                FadeIn(label(head, FG, 20).next_to(card.get_top(), DOWN, buff=0.18)),
                run_time=0.25,
            )

        # These are the unique minimum-cost assignments for one concrete
        # three-point squared-Euclidean example (documented in SOURCE_NOTES.md).
        perms = [[1, 0, 2], [2, 1, 0], [0, 2, 1]]
        datasets = [None, None, None]

        all_match_groups = VGroup()
        all_dot_groups = VGroup()
        for idx, (card, _dataset, perm, (c1, c2)) in enumerate(
            zip(cards, datasets, perms, cols)
        ):
            # Display indexed atoms in two columns; the geometry generated these assignments,
            # while the indices make composition consistency inspectable.
            lx, rx = card.get_center()[0] - 1.15, card.get_center()[0] + 1.15
            order_y = [0.78, 0.0, -0.78]
            left_dots = VGroup(
                *[Dot([lx, y, 0], radius=0.09, color=c1) for y in order_y]
            )
            right_dots = VGroup(
                *[Dot([rx, y, 0], radius=0.09, color=c2) for y in order_y]
            )
            # Explicit labels for each card.
            prefix_left = ["A", "B", "A"][idx]
            prefix_right = ["B", "C", "C"][idx]
            ids_left = VGroup(
                *[
                    formula(rf"{prefix_left}_{k}", 0.45, c1).next_to(
                        left_dots[k], LEFT, buff=0.08
                    )
                    for k in range(3)
                ]
            )
            ids_right = VGroup(
                *[
                    formula(rf"{prefix_right}_{k}", 0.45, c2).next_to(
                        right_dots[k], RIGHT, buff=0.08
                    )
                    for k in range(3)
                ]
            )
            lines = VGroup(
                *[
                    Line(left_dots[k], right_dots[perm[k]], color=GREEN, stroke_width=3)
                    for k in range(3)
                ]
            )
            all_dot_groups.add(VGroup(left_dots, right_dots, ids_left, ids_right))
            all_match_groups.add(lines)
        self.play(LaggedStart(*[FadeIn(g) for g in all_dot_groups], lag_ratio=0.15))
        self.play(
            LaggedStart(
                *[
                    LaggedStart(*[Create(l) for l in g], lag_ratio=0.12)
                    for g in all_match_groups
                ],
                lag_ratio=0.2,
            ),
            run_time=1.7,
        )

        chain = formula(r"A_0\to B_1\to C_1\quad\text{but}\quad A_0\to C_0", 0.78)
        chain.to_edge(DOWN, buff=0.72)
        cross = Cross(chain, stroke_color=RED, stroke_width=6)
        self.play(Write(chain), Create(cross))
        conclusion = label(
            "No single joint coupling J has all three plans as pairwise marginals",
            RED,
            22,
        ).to_edge(DOWN, buff=0.28)
        self.play(FadeIn(conclusion, shift=UP * 0.1))
        self.play(
            FadeIn(
                footer(
                    "Thesis §5.2–§5.3 — the portfolio theorem therefore retains one coherent joint exposure law"
                )
            )
        )
        self.wait(1.2)
