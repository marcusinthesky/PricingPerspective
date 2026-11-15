"""Named peer baskets for declared targets under three fixed peer definitions.

Each row is a coefficient of an operator scored in the factor-conditioned field
comparison (``p5_economic_value``): the target-anchored W2 barycentric field,
the median-bandwidth W2 RBF field, and the persistent co-mention field.  The
builders are shared with that stage through ``field_operators`` so an exhibit
cannot drift from the matrix that was scored.  The panel is descriptive only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

from pipeline.stages.papers.paper5.field_operators import (
    FieldOperatorIdentity,
    co_mention_operator,
    load_field_operator_inputs,
    rbf_median_operator,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from pipeline.stages.papers.paper5.field_operators import (
        FieldOperatorInputs,
        FieldOperatorPaths,
    )

#: Definition id -> economic-value model id scored in the field comparison.
PEER_DEFINITIONS: dict[str, str] = {
    "barycentric_w2": "w2",
    "rbf_w2_median": "rbf_median",
    "co_mention": "co_mentions",
}
PEER_DEFINITION_COLUMNS = (
    "target",
    "definition",
    "rank",
    "candidate",
    "weight",
    "remaining_mass",
    "remaining_count",
    "support_size",
    "same_sector",
    "same_industry",
    "target_sector",
    "target_industry",
    "candidate_sector",
    "candidate_industry",
)


@dataclass(frozen=True, slots=True)
class PeerDefinitionConfig:
    """Declared targets, basket length, and the transport-field identity."""

    targets: tuple[str, ...] = ("AMD", "HON", "JBHT", "XEL")
    top_k: int = 5
    identity: FieldOperatorIdentity = field(default_factory=FieldOperatorIdentity)

    def validate(self) -> None:
        """Reject empty or duplicated target lists before reading artifacts."""
        if not self.targets or len(set(self.targets)) != len(self.targets):
            message = "peer-definition targets must be non-empty and unique"
            raise ValueError(message)
        if self.top_k < 1:
            message = "peer-definition top_k must be positive"
            raise ValueError(message)


def peer_definition_operators(
    inputs: FieldOperatorInputs,
) -> dict[str, NDArray[np.float64]]:
    """Return the three scored operators keyed by peer-definition id."""
    return {
        "barycentric_w2": inputs.matrices["w2"],
        "rbf_w2_median": rbf_median_operator(inputs.squared_distances),
        "co_mention": co_mention_operator(inputs.co_mentions),
    }


def peer_definition_rows(
    inputs: FieldOperatorInputs,
    operators: dict[str, NDArray[np.float64]],
    config: PeerDefinitionConfig,
) -> pd.DataFrame:
    """Rank each target's positive coefficients; ties break by ticker."""
    missing = sorted(set(config.targets) - set(inputs.tickers))
    if missing:
        message = f"peer-definition targets are outside the universe: {missing}"
        raise ValueError(message)
    records: list[dict[str, object]] = []
    for target in config.targets:
        target_index = inputs.tickers.index(target)
        for definition, matrix in operators.items():
            row = matrix[target_index]
            support = [
                index
                for index in range(len(inputs.tickers))
                if index != target_index and row[index] > 0.0
            ]
            top = sorted(support, key=lambda i: (-row[i], inputs.tickers[i]))
            top = top[: config.top_k]
            remaining = float(row.sum() - row[top].sum())
            for rank, index in enumerate(top, start=1):
                candidate = inputs.tickers[index]
                records.append(
                    {
                        "target": target,
                        "definition": definition,
                        "rank": rank,
                        "candidate": candidate,
                        "weight": float(row[index]),
                        "remaining_mass": remaining,
                        "remaining_count": len(support) - len(top),
                        "support_size": len(support),
                        "same_sector": inputs.sector[target]
                        == inputs.sector[candidate],
                        "same_industry": inputs.industry[target]
                        == inputs.industry[candidate],
                        "target_sector": inputs.sector[target],
                        "target_industry": inputs.industry[target],
                        "candidate_sector": inputs.sector[candidate],
                        "candidate_industry": inputs.industry[candidate],
                    }
                )
    return pd.DataFrame.from_records(records, columns=list(PEER_DEFINITION_COLUMNS))


def caption_facts(rows: pd.DataFrame) -> dict[str, object]:
    """Summarize top peers, masses, and pairwise overlaps for caption checks."""
    targets: dict[str, object] = {}
    for target, target_rows in rows.groupby("target", sort=False):
        baskets: dict[str, object] = {}
        members: dict[str, set[str]] = {}
        for definition in PEER_DEFINITIONS:
            basket = target_rows.loc[target_rows["definition"] == definition]
            basket = basket.sort_values("rank")
            members[definition] = set(basket["candidate"])
            baskets[definition] = {
                "top": [
                    {
                        "rank": int(record["rank"]),
                        "candidate": str(record["candidate"]),
                        "weight": round(float(record["weight"]), 4),
                        "candidate_industry": str(record["candidate_industry"]),
                        "same_industry": bool(record["same_industry"]),
                    }
                    for record in basket.to_dict(orient="records")
                ],
                "top_mass": round(float(basket["weight"].sum()), 4),
                "remaining_mass": round(float(basket["remaining_mass"].iloc[0]), 4),
                "remaining_count": int(basket["remaining_count"].iloc[0]),
            }
        overlaps = {
            f"{left}&{right}": sorted(members[left] & members[right])
            for left, right in combinations(PEER_DEFINITIONS, 2)
        }
        overlaps["all_three"] = sorted(set.intersection(*members.values()))
        first = target_rows.iloc[0]
        targets[str(target)] = {
            "sector": str(first["target_sector"]),
            "industry": str(first["target_industry"]),
            "definitions": baskets,
            "overlap": overlaps,
            "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        }
    return {
        "schema_version": 1,
        "definitions": PEER_DEFINITIONS,
        "weights_rounded_to": 4,
        "classification_source": "data/universe.csv Sector/Industry",
        "descriptive_only": True,
        "targets": targets,
    }


def run_paper5_peer_definition_rows(
    paths: FieldOperatorPaths,
    config: PeerDefinitionConfig,
    output_dir: Path,
) -> dict[str, object]:
    """Write peer-definition rows and caption facts for the declared targets."""
    config.validate()
    inputs = load_field_operator_inputs(paths, config.identity)
    rows = peer_definition_rows(inputs, peer_definition_operators(inputs), config)
    facts = caption_facts(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output_dir / "rows.parquet", index=False)
    (output_dir / "caption_facts.json").write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return facts


__all__ = [
    "PEER_DEFINITIONS",
    "PEER_DEFINITION_COLUMNS",
    "PeerDefinitionConfig",
    "caption_facts",
    "peer_definition_operators",
    "peer_definition_rows",
    "run_paper5_peer_definition_rows",
]
