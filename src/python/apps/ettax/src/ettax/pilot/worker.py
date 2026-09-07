# SPDX-License-Identifier: Apache-2.0
"""One disposable JAX process used by :mod:`ettax.pilot.probe`."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import replace
from pathlib import Path


def main() -> None:
    """Compile and run one synthetic capacity candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    arguments = parser.parse_args()

    import flax
    import jax
    import optax

    from ettax.config import ExperimentConfig
    from ettax.pilot.config import PilotConfig
    from ettax.train import benchmark

    pilot = PilotConfig.from_toml(arguments.config)
    base = ExperimentConfig.from_toml(pilot.experiment_config)
    experiment = replace(
        base,
        data=replace(
            base.data,
            sequence_length=arguments.length,
            batch_size=arguments.batch_size,
        ),
        train=replace(
            base.train,
            token_budget=arguments.length * arguments.batch_size * arguments.steps,
        ),
    )
    result = benchmark(experiment, warmup=1, steps=arguments.steps)
    payload: dict[str, object] = {
        **result,
        "allocator": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        },
        "device": str(jax.devices()[0]),
        "environment": {
            "backend": jax.default_backend(),
            "flax": flax.__version__,
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "optax": optax.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "pid": os.getpid(),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
