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
    title_block,
    footer,
)


class CertificateConvexity(Scene):
    """Conditional negative definiteness of squared floors makes the risk objective convex on the simplex."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "12",
            "Geometry can certify convex optimization",
            "A checkable condition on the distance matrix — no return covariance required",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        left = panel(6.0, 4.15).move_to(LEFT * 3.45 + DOWN * 0.1)
        right = panel(5.8, 4.15).move_to(RIGHT * 3.6 + DOWN * 0.1)
        self.play(Create(left), Create(right))
        self.play(
            FadeIn(
                label("normalized risk-weight simplex", CYAN, 22).next_to(
                    left.get_top(), DOWN, buff=0.18
                )
            ),
            FadeIn(
                label("curvature test", GREEN, 22).next_to(
                    right.get_top(), DOWN, buff=0.18
                )
            ),
        )

        tri = Polygon(
            [-5.6, -1.4, 0],
            [-1.25, -1.4, 0],
            [-3.43, 1.45, 0],
            color=CYAN,
            stroke_width=3,
        )
        tri.set_fill(BLUE, opacity=0.08)
        verts = VGroup(
            formula(r"q_1=1", 0.52, BLUE).next_to(
                tri.get_vertices()[0], DOWN, buff=0.08
            ),
            formula(r"q_2=1", 0.52, YELLOW).next_to(
                tri.get_vertices()[1], DOWN, buff=0.08
            ),
            formula(r"q_3=1", 0.52, PURPLE).next_to(
                tri.get_vertices()[2], DOWN, buff=0.10
            ),
        )
        self.play(Create(tri), FadeIn(verts))

        contours = VGroup(
            *[
                Ellipse(
                    width=3.3 - 0.45 * k,
                    height=2.0 - 0.28 * k,
                    color=GREEN,
                    stroke_width=2,
                    stroke_opacity=0.75,
                ).move_to([-3.4, -0.25, 0])
                for k in range(4)
            ]
        )
        self.play(LaggedStart(*[Create(c) for c in contours], lag_ratio=0.13))
        start = Dot([-4.8, -0.95, 0], radius=0.1, color=RED)
        optimum = Dot([-3.4, -0.25, 0], radius=0.13, color=GREEN)
        path = VMobject(color=YELLOW, stroke_width=4).set_points_smoothly(
            [
                start.get_center(),
                [-4.2, -0.65, 0],
                [-3.8, -0.42, 0],
                optimum.get_center(),
            ]
        )
        self.play(
            FadeIn(start),
            Create(path),
            MoveAlongPath(start, path),
            FadeIn(optimum),
            run_time=1.6,
        )

        tangent = formula(r"\mathbf 1^\top u=0", 0.7, CYAN).move_to([3.5, 0.85, 0])
        cnd = formula(r"u^\top\ell^{\circ 2}u\le0", 0.85, GREEN).next_to(
            tangent, DOWN, buff=0.22
        )
        therefore = formula(
            r"C(q)\ \text{concave}\quad\Longrightarrow\quad1-C(q)\ \text{convex}",
            0.72,
            YELLOW,
        ).next_to(cnd, DOWN, buff=0.32)
        self.play(Write(tangent), Write(cnd), FadeIn(therefore, shift=UP * 0.1))

        check = (
            VGroup(
                label("Schoenberg check:", MUTED, 19),
                formula(r"-\tfrac12H\ell^{\circ2}H\succeq0", 0.72, FG),
            )
            .arrange(DOWN, buff=0.1)
            .next_to(right.get_bottom(), UP, buff=0.24)
        )
        self.play(FadeIn(check))
        self.play(
            FadeIn(
                footer(
                    "Thesis Theorem 5.5 — conditional negative definiteness is directly checkable from the observed floor matrix"
                )
            )
        )
        self.wait(1.2)
