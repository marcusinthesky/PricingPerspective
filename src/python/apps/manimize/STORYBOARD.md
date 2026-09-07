# Storyboard

The thirteen scenes are ordered as a progressive mathematical curriculum. Each included reference video lasts 7.5 seconds; native Manim timings follow the `self.play` and `self.wait` calls in each scene and may be longer.

## 01 — A firm as a probability law

**Purpose:** replace the familiar point-valued firm with a distribution-valued primitive.

1. Show one point for each firm.
2. Expand each point into an empirical cloud while retaining its centroid.
3. Contrast a broad unimodal cloud with a two-mode cloud.
4. State that a point mass is the degenerate special case of a probability law.

**Reader takeaway:** averaging can remove composition, spread, and multimodality.

## 02 — Couplings and quadratic transport

**Purpose:** define a coupling as a joint assignment rather than an abstract symbol.

1. Place two equal-mass clouds in parallel columns.
2. Draw a feasible but expensive assignment.
3. replace it with a lower-cost assignment.
4. Reveal the quadratic Wasserstein objective.

**Reader takeaway:** the marginals remain fixed while the joint arrangement varies.

## 03 — Polarization: distance becomes covariance

**Purpose:** explain why squared transport cost has a special second-moment role.

1. Draw two exposure vectors of fixed lengths.
2. Join their endpoints with a distance chord.
3. Display the polarization identity.
4. Move one vector closer while preserving its length.
5. Replace point geometry by optimization over couplings.

**Reader takeaway:** with marginal magnitudes fixed, minimizing expected squared distance maximizes expected inner product.

## 04 — A sharp covariance envelope

**Purpose:** move from one optimal coupling to the full attainable covariance interval.

1. Show several couplings of the same two marginal laws.
2. Mark the reflected lower endpoint, a realized interior coupling, and the Wasserstein ceiling.
3. Display the attainable interval.
4. Mark the transport-excess gap below the ceiling.

**Reader takeaway:** marginals identify an envelope, not the realized covariance or its sign.

## 05 — Transmission, distortion, and slack

**Purpose:** keep observable semantic geometry separate from latent factor-risk geometry.

1. Show two characteristic laws in information space.
2. Pass them through a common randomized carrier.
3. Display firm-specific slack halos in risk space.
4. Reveal the metric transfer bracket.

**Reader takeaway:** the financial conclusion is conditional on a declared bridge; embedding distance is not silently re-labelled as risk distance.

## 06 — Target-anchored barycentric reconstruction

**Purpose:** explain the two-stage construction of one interaction-field row.

1. Fix one target empirical law.
2. Align every candidate peer separately to the target.
3. Freeze those correspondences.
4. Combine aligned peer positions using one common simplex vector.
5. Show the reconstructed target and fitted weights.

**Reader takeaway:** the row is a joint reconstruction solution, not a scalar transform of pairwise distance.

## 07 — A symmetric metric can yield a directed field

**Purpose:** separate symmetric separation from asymmetric reconstruction usefulness.

1. Display a symmetric triangle of pairwise distances.
2. Replace undirected links with target-specific weighted arrows.
3. Compare opposite directions between the same two firms.
4. State that direction records reconstruction relevance.

**Reader takeaway:** `W2(C_i,C_j)=W2(C_j,C_i)` is compatible with `W^flat_ij != W^flat_ji`.

## 08 — Quadratic adjustment implies spatial closure

**Purpose:** derive, rather than posit, the spatial exposure equation.

1. Place stand-alone exposure and peer-average exposure on a line.
2. Locate the peer-adjusted exposure between them.
3. Label the two quadratic penalties.
4. Show the map from relative penalty `lambda` to feedback `rho`.
5. Reveal the spatial closure and normalized multiplier.

**Reader takeaway:** the spatial coefficient has a conditional adjustment interpretation at the exposure layer.

## 09 — Pairwise plans need not form one joint law

**Purpose:** make the portfolio compatibility problem concrete.

1. Show independently optimal assignments for A–B, B–C, and A–C.
2. Follow one atom through A to B to C.
3. Compare the implied A–C pairing with the independently optimal A–C plan.
4. Mark the contradiction.

**Reader takeaway:** separately optimal pairwise couplings cannot automatically be stacked into a positive-semidefinite portfolio risk system.

## 10 — Multi-firm transport dispersion

**Purpose:** introduce the sharp coherent aggregation object.

1. Draw a realization that joins all firm laws simultaneously.
2. Display weighted total pairwise dispersion under a common coupling.
3. Morph to the equivalent free-centre representation.
4. Hold portfolio weights fixed while the common coupling or centre varies.

**Reader takeaway:** this free-centre barycentre differs from Essay II's fixed-target reconstruction.

## 11 — Portfolio variance as alignment minus dispersion

**Purpose:** turn weighted polarization into a one-sided risk certificate.

1. Show perfectly aligned exposure vectors and the corresponding risk benchmark.
2. Introduce certified pairwise separation floors.
3. Rotate the vectors apart.
4. Visually cut the certificate credit from the admissible risk bar.
5. State the portfolio variance bound.

**Reader takeaway:** observed geometry rules out part of worst-case alignment without estimating a covariance matrix.

## 12 — Geometry can certify convex optimization

**Purpose:** connect the floor matrix to portfolio curvature.

1. Show a three-asset normalized risk-weight simplex.
2. Draw nested objective contours and an optimizer path.
3. State the tangent-space condition `1^T u = 0`.
4. Reveal conditional negative definiteness.
5. Show the double-centred Schoenberg check and convex conclusion.

**Reader takeaway:** convexity can be checked directly from the observed geometric matrix rather than assumed from returns.

## 13 — Three distinct optimization layers

**Purpose:** prevent the shared language of couplings, barycentric geometry, and weights from collapsing three different problems into one.

1. Hold two exposure marginals fixed and vary a pairwise coupling.
2. Hold one target and its alignments fixed and vary simplex peer coordinates.
3. Hold portfolio weights fixed inside multi-firm dispersion and vary a coherent coupling or free centre.
4. Separate the fixed-`q` inner problem from the outer portfolio choice over `q`.

**Reader takeaway:** the same geometric vocabulary supports three different financial objects because each optimization fixes and varies different quantities.
