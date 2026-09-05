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
    centroid_marker,
    label,
    formula,
    title_block,
    footer,
)


class FirmAsDistribution(Scene):
    """A point-valued firm is a degenerate distribution; a cloud retains composition."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "01",
            "A firm as a probability law",
            "Retaining spread, multimodality and internal composition",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        left = panel(5.7, 3.8).move_to(LEFT * 3.25 + DOWN * 0.25)
        right = panel(5.7, 3.8).move_to(RIGHT * 3.25 + DOWN * 0.25)
        la = label("Firm A", BLUE, 25).next_to(left.get_top(), DOWN, buff=0.22)
        lb = label("Firm B", YELLOW, 25).next_to(right.get_top(), DOWN, buff=0.22)
        self.play(Create(left), Create(right), FadeIn(la), FadeIn(lb))

        dot_a = Dot(left.get_center(), radius=0.14, color=BLUE)
        dot_b = Dot(right.get_center(), radius=0.14, color=YELLOW)
        point_caption = label("Point summaries", MUTED, 22).next_to(
            VGroup(left, right), DOWN, buff=0.22
        )
        self.play(GrowFromCenter(dot_a), GrowFromCenter(dot_b), FadeIn(point_caption))
        same = label("Both centroids are identical", RED, 22).move_to(DOWN * 2.45)
        self.play(Transform(point_caption, same))
        self.wait(0.5)

        a_pts = [
            (-3.65, -0.5),
            (-3.45, 0.05),
            (-3.35, -0.7),
            (-3.15, 0.25),
            (-2.95, -0.35),
            (-3.55, 0.55),
            (-3.05, 0.7),
            (-2.75, 0.15),
            (-2.7, -0.65),
            (-3.2, -0.1),
        ]
        b_pts = [
            (2.25, -0.55),
            (2.45, -0.15),
            (2.65, 0.35),
            (2.45, 0.65),
            (2.7, -0.8),
            (3.8, -0.55),
            (4.0, -0.1),
            (4.2, 0.35),
            (4.05, 0.75),
            (3.75, -0.85),
        ]
        ca = cloud(a_pts, BLUE, radius=0.09)
        cb = cloud(b_pts, YELLOW, radius=0.09)
        ma = centroid_marker(left.get_center(), GREEN)
        mb = centroid_marker(right.get_center(), GREEN)
        self.play(
            FadeOut(dot_a),
            FadeOut(dot_b),
            LaggedStart(*[GrowFromCenter(d) for d in ca], lag_ratio=0.04),
            LaggedStart(*[GrowFromCenter(d) for d in cb], lag_ratio=0.04),
            run_time=1.6,
        )
        self.play(Create(ma), Create(mb), FadeOut(point_caption))

        a_desc = label("one broad mode", BLUE, 21).next_to(
            left.get_bottom(), UP, buff=0.18
        )
        b_desc = label("two separated modes", YELLOW, 21).next_to(
            right.get_bottom(), UP, buff=0.18
        )
        self.play(FadeIn(a_desc), FadeIn(b_desc))

        equation = formula(
            r"C_i\in\mathcal P_2(\Omega),\qquad C_i=\delta_{x_i}\ \text{is the point-valued special case}",
            0.68,
        )
        equation.to_edge(DOWN, buff=0.48)
        self.play(Write(equation))
        self.play(
            FadeIn(footer("Thesis §1.2.1 — the distribution as a modelling primitive"))
        )
        self.wait(1.2)
