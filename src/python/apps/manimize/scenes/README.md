# Manim scene sources

Each file defines one standalone `Scene` subclass and imports shared styling from `../theme.py`.

Run a scene from the project root, for example:

```bash
uv run --project . --extra native python -m manim -qm scenes/06_target_anchored_reconstruction.py TargetAnchoredReconstruction
```

The scenes use synthetic low-dimensional illustrations to explain the mathematics. They do not load or reproduce the manuscripts' empirical article clouds, distance matrix, return panel, or firm-level outputs.
