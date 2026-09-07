"""Small, stable helpers shared by the H1 frontier / OOS-coverage stages.

Lifted out of the large ``analysis`` and ``paper3_empirical`` modules so that
``h1_frontier`` / ``h1_oos_coverage`` (and ``returns_loadings``) depend only on this
single-purpose module. Unrelated edits to those big modules no longer invalidate the
H1 stages. See the pipeline architecture's dependency-scoping section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

Float = NDArray[np.float64]


def _load_typed_distances(
    artifact_dir: Path,
    *,
    expected_identity: dict[str, object] | None = None,
) -> tuple[list[str], np.ndarray, dict[str, object]]:
    """Load a self-identifying typed distance matrix and its governed metadata."""
    frame, summary = read_typed_distance_artifact(
        artifact_dir,
        expected_identity=expected_identity or {},
    )
    item_ids = summary["item_ids"]
    if not isinstance(item_ids, list):
        message = "typed distance summary item_ids must be a list"
        raise TypeError(message)
    tickers = [str(item) for item in item_ids]
    matrix = (
        frame["value"].to_numpy(dtype=np.float64).reshape(len(tickers), len(tickers))
    )
    return tickers, matrix, summary


def _load_energy_distances(
    artifact_dir: Path,
    *,
    provider_id: str,
    representation_id: str,
    distance_id: str,
) -> tuple[list[str], np.ndarray]:
    """Load an explicitly identified energy functional artifact."""
    tickers, matrix, _summary = _load_typed_distances(
        artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
        },
    )
    return tickers, matrix


def load_energy_functional_pairs(
    artifact_dir: Path,
    *,
    provider_id: str,
    representation_id: str,
    distance_id: str,
) -> pd.DataFrame:
    """Adapt the typed full matrix to legacy paper-local pair column names."""
    frame, _summary = read_typed_distance_artifact(
        artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
        },
    )
    return frame.rename(
        columns={"item_i": "symbol1", "item_j": "symbol2", "value": "energy_distance"}
    )


def _cluster_assignments(
    squared_distances: Float, n_clusters: int, seed: int = 0
) -> NDArray[np.int64]:
    """Assign the ``C(n,2)`` pairs to ``n_clusters`` via k-means on pair features.

    Each pair ``(i, j)`` is featured by its squared energy distance ``D²_ij``;
    pairs with similar distances cluster together, giving portfolio-level
    moments whose HAC is invertible.

    Args:
        squared_distances: Squared energy-distance matrix, shape ``(n, n)``.
        n_clusters: Target number of clusters.
        seed: RNG seed for k-means.

    Returns:
        Cluster index per pair, shape ``(P,)``.

    """
    n = squared_distances.shape[0]
    iu = np.triu_indices(n, k=1)
    feats = squared_distances[iu][:, None]
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(feats).astype(np.int64)


def _to_plain(obj: object) -> object:
    """Recursively coerce numpy/JAX scalars and arrays to plain Python for YAML.

    JAX leaves reach here because jcor's traced kernels return them directly -
    ``jcor.optimize.psd.repair_diagnostics``, for one, is now a traced function
    whose fields are 0-d arrays. A JAX array is normalised to NumPy and then
    falls through the existing branches rather than recursing, so the two
    array planes share one coercion path.
    """
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, jax.Array):
        obj = np.asarray(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        arr = cast("np.ndarray[Any, np.dtype[Any]]", obj)
        if arr.ndim == 0:
            return _to_plain(arr.item())
        return [_to_plain(v) for v in arr]
    return obj
