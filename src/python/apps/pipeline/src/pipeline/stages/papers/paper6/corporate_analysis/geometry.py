"""Disposable shared squared-distance costs and independent OT qualification."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from hotelling.dynamics.artifacts import file_hash
from scipy.optimize import linprog
from scipy.sparse import eye, kron, vstack
from scipy.spatial.distance import cdist

from pipeline._kernels.paper6.corporate_analysis import (
    DIMENSIONS,
    OBJECTIVE_TOLERANCE,
    SUPPORTS,
    YEARS,
    exact_transport,
    normalized,
    point_masses,
)
from pipeline.stages.papers.paper6.artifacts import read, write

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline._kernels.paper6.corporate_analysis import Cloud, CloudKey, FloatArray


def pairs(n: int) -> list[tuple[int, int, int]]:
    """Reuse symmetric origin transport while retaining directed later years."""
    return [(0, i, j) for i in range(n) for j in range(i + 1, n)] + [
        (t, i, j)
        for t in range(1, len(YEARS))
        for i in range(n)
        for j in range(n)
        if i != j
    ]


class CostCache:
    """Read-only flat memmaps shared by bounded workers."""

    def __init__(self, directory: Path, counts: np.ndarray) -> None:
        """Index all expanded target/origin cost blocks."""
        self.directory = directory
        self.n = counts.shape[1]
        self.counts = counts
        self.entries = pairs(self.n)
        lengths = [counts[t, i] * counts[0, j] for t, i, j in self.entries]
        self.offsets = np.r_[0, np.cumsum(lengths)]
        self.lookup = {key: index for index, key in enumerate(self.entries)}
        self.arrays: dict[int, FloatArray] = {}

    def cost(self, dimension: int, t: int, i: int, j: int) -> FloatArray:
        """Use one origin entry for both orientations and every role."""
        if dimension not in self.arrays:
            self.arrays[dimension] = np.load(
                self.directory / f"d{dimension}.npy", mmap_mode="r", allow_pickle=False
            )
        transpose = t == 0 and i > j
        key = (t, j, i) if transpose else (t, i, j)
        index = self.lookup[key]
        _, left, right = key
        value = self.arrays[dimension][
            self.offsets[index] : self.offsets[index + 1]
        ].reshape(self.counts[t, left], self.counts[0, right])
        return value.T if transpose else value


def build_cache(
    directory: Path, identity: str, clouds: dict[CloudKey, Cloud], n: int
) -> tuple[CostCache, dict[str, Any]]:
    """Admit only complete float64, outcome-blind cost arrays."""
    directory.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([[len(clouds[i, y]["core"]) for i in range(n)] for y in YEARS])
    cache = CostCache(directory, counts)
    result: dict[str, Any] = {
        "identity": identity,
        "counts": counts.tolist(),
        "problems_per_representation": len(cache.entries),
        "cost_entries_per_representation": int(cache.offsets[-1]),
        "dimensions": {},
    }
    for dimension in DIMENSIONS:
        path = directory / f"d{dimension}.npy"
        meta = directory / f"d{dimension}.json"
        if path.exists() and meta.exists():
            saved = read(meta)
            if (
                saved["identity"] != identity
                or saved["counts"] != counts.tolist()
                or saved["sha256"] != file_hash(path)
            ):
                msg = "corporate cost cache identity changed"
                raise ValueError(msg)
            result["dimensions"][str(dimension)] = saved
            continue
        if path.exists() or meta.exists():
            msg = "incomplete admitted corporate cost cache"
            raise ValueError(msg)
        points = {
            key: normalized(cloud["points"], dimension) for key, cloud in clouds.items()
        }
        temporary = path.with_suffix(".partial.npy")
        values = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float64, shape=(int(cache.offsets[-1]),)
        )
        started = time.perf_counter()
        for index, (t, i, j) in enumerate(cache.entries):
            cost = cdist(points[i, YEARS[t]], points[j, YEARS[0]], "sqeuclidean")
            values[cache.offsets[index] : cache.offsets[index + 1]] = cost.ravel()
        values.flush()
        del values
        temporary.replace(path)
        saved = {
            "identity": identity,
            "counts": counts.tolist(),
            "dimension": dimension,
            "sha256": file_hash(path),
            "seconds": time.perf_counter() - started,
            "bytes": path.stat().st_size,
        }
        write(meta, saved)
        result["dimensions"][str(dimension)] = saved
    return cache, result


def distances(
    cache: CostCache, dimension: int, masses: dict[CloudKey, FloatArray]
) -> FloatArray:
    """Evaluate fixed-origin squared Wasserstein distances for every nonself pair."""
    result = np.full((len(YEARS), cache.n, cache.n), np.nan)
    for t, i, j in cache.entries:
        result[t, i, j] = exact_transport(
            cache.cost(dimension, t, i, j), masses[i, YEARS[t]], masses[j, YEARS[0]]
        )
        if t == 0:
            result[t, j, i] = result[t, i, j]
    return result


def certificates(
    cache: CostCache, clouds: dict[CloudKey, Cloud]
) -> list[dict[str, Any]]:
    """Compare deterministic unequal-support endpoints with independent HiGHS LP."""
    indices = sorted(
        set(np.linspace(0, len(cache.entries) - 1, 8, dtype=int).tolist())
        | {int(np.argmax(np.diff(cache.offsets)))}
    )
    rows = []
    for dimension in DIMENSIONS:
        for support in SUPPORTS:
            masses = point_masses(clouds, support)
            for index in indices:
                t, i, j = cache.entries[index]
                a, b = masses[i, YEARS[t]], masses[j, YEARS[0]]
                left, right = np.flatnonzero(a), np.flatnonzero(b)
                cost = np.asarray(cache.cost(dimension, t, i, j)[np.ix_(left, right)])
                a, b = a[left], b[right]
                n, m = cost.shape
                constraints = vstack(
                    (kron(eye(n), np.ones((1, m))), kron(np.ones((1, n)), eye(m))),
                    format="csr",
                )
                solved = linprog(
                    cost.ravel(),
                    A_eq=constraints,
                    b_eq=np.r_[a, b],
                    bounds=(0, None),
                    method="highs",
                )
                if not solved.success:
                    msg = (
                        "independent corporate transport certificate failed: "
                        f"{solved.message}"
                    )
                    raise ValueError(msg)
                value = exact_transport(cost, a, b, certify=True)
                if abs(value - solved.fun) > OBJECTIVE_TOLERANCE * max(1, abs(value)):
                    msg = "POT and independent HiGHS corporate transports disagree"
                    raise ValueError(msg)
                rows.append(
                    {
                        "dimension": dimension,
                        "support": support,
                        "pair": [t, i, j],
                        "shape": list(cost.shape),
                        "pot_objective": value,
                        "highs_objective": float(solved.fun),
                        "absolute_difference": abs(value - float(solved.fun)),
                    }
                )
    return rows
