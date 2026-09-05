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


class TargetAnchoredReconstruction(Scene):
    """Pairwise transport establishes correspondence; one simplex problem chooses joint peer coordinates."""

    def construct(self):
        self.camera.background_color = BG
        title = title_block(
            "06",
            "Target-anchored barycentric reconstruction",
            "From article-level correspondence to a row of peer weights",
        )
        self.play(FadeIn(title, shift=DOWN * 0.15))

        target_box = panel(3.1, 3.6).move_to(LEFT * 4.75 + DOWN * 0.15)
        align_box = panel(4.2, 3.6).move_to(LEFT * 0.9 + DOWN * 0.15)
        recon_box = panel(4.2, 3.6).move_to(RIGHT * 3.7 + DOWN * 0.15)
        self.play(Create(target_box), Create(align_box), Create(recon_box))
        self.play(
            FadeIn(
                label("A. fixed target", BLUE, 21).next_to(
                    target_box.get_top(), DOWN, buff=0.18
                )
            ),
            FadeIn(
                label("B. target-to-peer alignments", CYAN, 21).next_to(
                    align_box.get_top(), DOWN, buff=0.18
                )
            ),
            FadeIn(
                label("C. convex reconstruction", GREEN, 21).next_to(
                    recon_box.get_top(), DOWN, buff=0.18
                )
            ),
        )

        target_pts = [
            (-5.25, 0.75),
            (-4.55, 0.55),
            (-5.1, -0.05),
            (-4.45, -0.45),
            (-5.15, -0.9),
        ]
        target = cloud(target_pts, BLUE, 0.095)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in target], lag_ratio=0.08))
        ti = formula(r"\widehat C_i", 0.7, BLUE).next_to(target, DOWN, buff=0.15)
        self.play(FadeIn(ti))

        aligned_sets = [
            [(-1.8, 0.75), (-1.35, 0.55), (-1.65, -0.05), (-1.2, -0.45), (-1.75, -0.9)],
            [(-0.7, 0.75), (-0.25, 0.55), (-0.55, -0.05), (-0.1, -0.45), (-0.65, -0.9)],
            [(0.4, 0.75), (0.85, 0.55), (0.55, -0.05), (1.0, -0.45), (0.45, -0.9)],
        ]
        cols = [YELLOW, ORANGE, PURPLE]
        peers = VGroup(*[cloud(pts, col, 0.07) for pts, col in zip(aligned_sets, cols)])
        peer_labels = VGroup(
            *[
                formula(rf"j_{k + 1}", 0.6, col).next_to(peers[k], DOWN, buff=0.12)
                for k, col in enumerate(cols)
            ]
        )
        self.play(
            LaggedStart(*[FadeIn(p, shift=UP * 0.12) for p in peers], lag_ratio=0.18),
            FadeIn(peer_labels),
        )

        match_arrows = VGroup()
        for k, peer in enumerate(peers):
            for m in range(5):
                match_arrows.add(
                    DashedLine(
                        target[m].get_center(),
                        peer[m].get_center(),
                        color=cols[k],
                        stroke_width=1.2,
                        stroke_opacity=0.45,
                    )
                )
        self.play(
            LaggedStart(*[Create(a) for a in match_arrows], lag_ratio=0.015),
            run_time=1.4,
        )
        self.play(FadeOut(match_arrows))

        weights = [0.42, 0.35, 0.23]
        recon_pts = []
        for m in range(5):
            v = sum(weights[k] * np.array(aligned_sets[k][m]) for k in range(3))
            # Move the reconstructed pattern into the final panel.
            recon_pts.append((float(v[0] + 4.2), float(v[1])))
        recon = cloud(recon_pts, GREEN, 0.095)
        target_copy = target.copy().shift(RIGHT * 8.45).set_opacity(0.35)
        self.play(
            FadeIn(target_copy),
            LaggedStart(*[GrowFromCenter(d) for d in recon], lag_ratio=0.08),
        )
        weights_row = VGroup(
            *[pill(f"{w:.2f}", col) for w, col in zip(weights, cols)]
        ).arrange(RIGHT, buff=0.14)
        weights_row.next_to(recon_box.get_bottom(), UP, buff=0.2)
        self.play(FadeIn(weights_row, shift=UP * 0.1))

        eq = formula(
            r"w_i\in\arg\min_{w\in\Delta_{-i}}\frac1M\sum_m\left\|x_{im}-\sum_{j\ne i}w_j y_{ijm}\right\|^2",
            0.64,
        )
        eq.to_edge(DOWN, buff=0.48)
        self.play(Write(eq))
        self.play(
            FadeIn(
                footer(
                    "Thesis Eqs. (4.11)–(4.12) — align each peer to the fixed target, then solve one joint simplex problem"
                )
            )
        )
        self.wait(1.2)
