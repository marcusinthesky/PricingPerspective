"""S2 — ground metrics: ``d: X x X -> R`` on points.

Position
--------
rank 2

Consumes
--------
raw arrays of points

Produces
--------
a pointwise distance matrix; each callable declares its axioms

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

This package is a t46 skeleton: modules land here via the t46.4-t46.9
migrations. Re-exports are appended per-owner, in the region marked below.
"""

from __future__ import annotations

__all__: list[str] = []

# --- t46 migration re-exports; each task appends only to its own region ---

# --- t46.5 (metrics) ---
from jcor.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    PROJECTIVE_ANGULAR,
    RAY_ANGULAR,
    RAY_CHORD,
    SPHERE_ANGULAR,
    SPHERE_CHORD,
    AngularLaw,
    CosineLaw,
    EuclideanLaw,
    GroundDistance,
    GroundDistanceFunction,
    GroundLawDeclaration,
    NormalizedEuclideanLaw,
    ProjectiveAngularLaw,
    SphereAngularLaw,
    angular_distance,
    cdist,
    checked_cloud_pdist,
    cosine_distance,
    declare_ground_distance,
    euclidean_distance,
    normalized_euclidean_distance,
    pairwise_distances,
    pdist,
    projective_angular_distance,
    sphere_angular_distance,
    sphere_chord_distance,
)
from jcor.ground.metrics import (
    DECLARED_AXIOMS as METRICS_DECLARED_AXIOMS,
)
from jcor.ground.metrics import (
    DECLARED_BRANDS as METRICS_DECLARED_BRANDS,
)

__all__ += [
    "ANGULAR",
    "COSINE",
    "EUCLIDEAN",
    "METRICS_DECLARED_AXIOMS",
    "METRICS_DECLARED_BRANDS",
    "NORMALIZED_EUCLIDEAN",
    "PROJECTIVE_ANGULAR",
    "RAY_ANGULAR",
    "RAY_CHORD",
    "SPHERE_ANGULAR",
    "SPHERE_CHORD",
    "AngularLaw",
    "CosineLaw",
    "EuclideanLaw",
    "GroundDistance",
    "GroundDistanceFunction",
    "GroundLawDeclaration",
    "NormalizedEuclideanLaw",
    "ProjectiveAngularLaw",
    "SphereAngularLaw",
    "angular_distance",
    "cdist",
    "checked_cloud_pdist",
    "cosine_distance",
    "declare_ground_distance",
    "euclidean_distance",
    "normalized_euclidean_distance",
    "pairwise_distances",
    "pdist",
    "projective_angular_distance",
    "sphere_angular_distance",
    "sphere_chord_distance",
]

# --- t55.2 (serialized configuration boundary) ---
from jcor.ground.config import (  # noqa: E402
    AngularDistanceConfig,
    BuiltinGroundDistance,
    CosineDistanceConfig,
    EuclideanDistanceConfig,
    GroundDistanceConfig,
    GroundDistanceName,
    GroundDistanceSelection,
    InvalidGroundDistanceConfigError,
    NormalizedEuclideanDistanceConfig,
    is_euclidean,
    parse_ground_distance,
    parse_ground_distance_name,
    resolve_ground_distance,
)

__all__ += [
    "AngularDistanceConfig",
    "BuiltinGroundDistance",
    "CosineDistanceConfig",
    "EuclideanDistanceConfig",
    "GroundDistanceConfig",
    "GroundDistanceName",
    "GroundDistanceSelection",
    "InvalidGroundDistanceConfigError",
    "NormalizedEuclideanDistanceConfig",
    "is_euclidean",
    "parse_ground_distance",
    "parse_ground_distance_name",
    "resolve_ground_distance",
]

# --- t50 (similarities) ---
from jcor.ground.similarities import (  # noqa: E402
    cosine_gram,
    distance_induced_gram,
    gram,
    linear_gram,
    rbf_gram,
)

__all__ += [
    "cosine_gram",
    "distance_induced_gram",
    "gram",
    "linear_gram",
    "rbf_gram",
]
