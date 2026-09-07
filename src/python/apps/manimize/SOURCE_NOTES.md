# Manuscript source notes

Source: **Marcus Gawronsky, _Distributional Information Geometry for Financial Dependence_, University of Cape Town, August 2026.** Page references below distinguish the printed page from the 1-indexed PDF page.

The animations are explanatory abstractions. They preserve the source's terminology and claims but use small synthetic point clouds rather than the empirical article embeddings or return data.

| Scene | Manuscript anchor | Printed page | PDF page | Source concept retained |
|---:|---|---:|---:|---|
| 01 | §1.2.1, Eq. (1.1) | 3 | 21 | `C_i` is a probability law in `P_2(Omega)`; a point mass is the point-valued special case. |
| 02 | §1.2.2, Eq. (1.2) | 3 | 21 | `W_2^2` is the least expected squared displacement over couplings. |
| 03 | Eq. (1.3); Lemma 3.1 / Eq. (3.7) | 4; 21 | 22; 39 | Polarization converts squared distance into an inner product; minimizing quadratic transport cost maximizes attainable covariance over couplings. |
| 04 | Theorems 3.1–3.2, Eqs. (3.10)–(3.12) | 22–23 | 40–41 | The maximum-covariance coupling supplies the ceiling, reflection supplies the floor, and transport excess records the gap below the ceiling. |
| 05 | Corollary 3.1, Eqs. (3.15)–(3.18) | 24 | 42 | A common bi-Lipschitz carrier plus additive slack transfers an observable distance into a latent distance bracket and then a conditional covariance-ceiling interval. |
| 06 | Definition 4.4, Eqs. (4.11)–(4.12); Figure 4.2 | 48–49 | 66–67 | Each peer is aligned to a fixed target first; then one simplex problem chooses joint distributional spanning weights. |
| 07 | §4.4, discussion after Definition 4.4 | 49 | 67 | `W^flat` is generally asymmetric because alignments and reconstruction are target-specific, even though pairwise `W_2` is symmetric. |
| 08 | Definition 4.2; Theorems 4.1–4.2, Eqs. (4.3)–(4.7) | 44–46 | 62–64 | A quadratic stand-alone-versus-peer objective yields `B = rho W B + (1-rho) xi` and the normalized spatial multiplier. |
| 09 | §5.2 and §5.3 opening | 71–73 | 89–91 | Pairwise optimal couplings need not coexist as marginals of one coherent joint exposure law. |
| 10 | Definition 5.1, Eqs. (5.9)–(5.11) | 74–75 | 92–93 | Weighted multi-firm dispersion varies one common coupling; in Hilbert space it has a free-centre Wasserstein-barycentre representation. |
| 11 | Eq. (5.7); Theorem 5.1 / Eq. (5.8) | 74 | 92 | Weighted Hilbert polarization gives a perfect-alignment benchmark minus certified cross-firm dispersion. |
| 12 | Theorem 5.5 and Schoenberg criterion | 79 | 97 | Conditional negative definiteness of the squared floor matrix makes `1-C(q)` convex on the simplex. |
| 13 | §2.2, “Separation, reconstruction, and dispersion” | 12 | 30 | The three essays keep distinct the variables optimized in pairwise coupling, target reconstruction, and fixed-weight multi-firm dispersion. |

## Three optimizations that must not be conflated

1. **Essay I:** hold two marginal laws fixed and vary their pairwise coupling.
2. **Essay II:** hold one target and its pairwise alignments fixed and vary simplex peer coordinates.
3. **Essay III:** hold portfolio weights fixed inside the dispersion functional and vary one coherent common coupling, or equivalently a free centre law; the outer portfolio problem varies weights only afterwards.

## Maintained interpretation boundaries

- The article-embedding law is a proxy for a latent characteristic law.
- Information distance does not identify an exposure law, covariance, or expected return.
- Direction in the interaction field records target-specific reconstruction relevance, not causal influence.
- The spatial adjustment interpretation requires the maintained exposure objective and return bridge.
- The portfolio result is an upper-risk certificate under one coherent joint exposure law, not covariance-free recovery of the global minimum-variance portfolio.

## Concrete construction used in Scene 09

Scene 09 uses the following synthetic three-point clouds in two-dimensional Euclidean space:

```text
A = [(-0.11,-0.84), (-0.82, 0.65), ( 0.74, 0.54)]
B = [(-0.67, 0.23), ( 0.12, 0.22), ( 0.87, 0.22)]
C = [( 0.68, 0.07), ( 0.29, 0.63), (-1.46,-0.32)]
```

Under equal masses and squared Euclidean assignment cost, the unique optimal permutations are:

```text
A -> B : [1, 0, 2]
B -> C : [2, 1, 0]
A -> C : [0, 2, 1]
```

Thus `A_0 -> B_1 -> C_1`, whereas the direct optimum requires `A_0 -> C_0`. The example is pedagogical and is not drawn from the manuscripts' empirical sample.
