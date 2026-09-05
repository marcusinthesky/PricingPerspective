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
    PURPLE,
    panel,
    cloud,
    label,
    formula,
    title_block,
    footer,
)


class CovarianceEnvelope(Scene):
    """Fixed exposure marginals define an attainable covariance interval, not a point."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "04",
            "A sharp covariance envelope",
            "Vary the coupling while keeping both exposure laws fixed",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        top = panel(12.0, 2.5).move_to(UP * 0.65)
        self.play(Create(top))
        left_pts = [(-5.0, 1.0), (-5.0, 0.45), (-5.0, -0.1)]
        right_pts = [(5.0, 1.0), (5.0, 0.45), (5.0, -0.1)]
        A = cloud(left_pts, BLUE, 0.09)
        B = cloud(right_pts, YELLOW, 0.09)
        self.play(FadeIn(A), FadeIn(B))

        perms = [[2, 0, 1], [0, 2, 1], [0, 1, 2]]
        colors = [RED, CYAN, GREEN]
        values = [r"\kappa^-_{ij}", r"\kappa_\pi", r"\bar\kappa_{ij}"]
        xs = [-2.4, 0.0, 2.4]
        cards = VGroup()
        for c, perm, col, val, x in zip(range(3), perms, colors, values, xs):
            card = panel(2.0, 1.45).move_to([x, 0.65, 0])
            lmini = [
                card.get_left() + RIGHT * 0.25 + UP * 0.4,
                card.get_left() + RIGHT * 0.25,
                card.get_left() + RIGHT * 0.25 + DOWN * 0.4,
            ]
            rmini = [
                card.get_right() + LEFT * 0.25 + UP * 0.4,
                card.get_right() + LEFT * 0.25,
                card.get_right() + LEFT * 0.25 + DOWN * 0.4,
            ]
            dots = VGroup(
                *[Dot(p, radius=0.045, color=BLUE) for p in lmini],
                *[Dot(p, radius=0.045, color=YELLOW) for p in rmini],
            )
            lines = VGroup(
                *[
                    Line(lmini[k], rmini[perm[k]], color=col, stroke_width=2.2)
                    for k in range(3)
                ]
            )
            t = formula(val, 0.6, col).next_to(card, DOWN, buff=0.06)
            cards.add(VGroup(card, lines, dots, t))
        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in cards], lag_ratio=0.18)
        )

        numberline = NumberLine(
            x_range=[-1, 1, 0.5],
            length=8.3,
            include_ticks=False,
            include_numbers=False,
            color=MUTED,
        )
        numberline.move_to(DOWN * 1.3)
        low = Dot(numberline.n2p(-0.75), color=RED, radius=0.105)
        mid = Dot(numberline.n2p(0.05), color=CYAN, radius=0.09)
        high = Dot(numberline.n2p(0.75), color=GREEN, radius=0.105)
        ltxt = formula(r"\kappa^-_{ij}", 0.65, RED).next_to(low, DOWN, buff=0.13)
        mtxt = formula(r"\kappa_\pi", 0.65, CYAN).next_to(mid, UP, buff=0.13)
        htxt = formula(r"\bar\kappa_{ij}", 0.65, GREEN).next_to(high, DOWN, buff=0.13)
        interval = Line(
            low.get_center(), high.get_center(), color=PURPLE, stroke_width=7
        )
        self.play(
            Create(numberline),
            Create(interval),
            FadeIn(low),
            FadeIn(mid),
            FadeIn(high),
            FadeIn(ltxt),
            FadeIn(mtxt),
            FadeIn(htxt),
        )

        gap = BraceBetweenPoints(mid.get_center(), high.get_center(), UP, color=YELLOW)
        gap_label = formula(r"\tfrac12\Delta_\pi", 0.65, YELLOW).next_to(
            gap, UP, buff=0.08
        )
        eq = formula(
            r"\kappa_\pi=\bar\kappa_{ij}-\tfrac12\Delta_\pi,\qquad \Delta_\pi\ge0", 0.78
        )
        eq.to_edge(DOWN, buff=0.46)
        self.play(GrowFromCenter(gap), FadeIn(gap_label), Write(eq))
        self.play(
            FadeIn(
                footer(
                    "Thesis Theorems 3.1–3.2 — the marginals identify a feasible interval, not the realized coupling"
                )
            )
        )
        self.wait(1.2)
