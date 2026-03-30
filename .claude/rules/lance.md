---
paths:
  - "notebooks/lance/**"
---

# Lance Notebook Rules

## PyPI package name is `pylance`, not `lance`

The Lance columnar format library is published on PyPI as `pylance` (which provides `import lance`).
The PyPI package named `lance` is an unrelated project.
In PEP 723 dependencies, always write `pylance>=X.Y`.

## DataLoader must use `num_workers=0`

Lance datasets accessed via `ds.take()` segfault when called from forked worker subprocesses.
This affects both macOS (fork safety) and Linux/Colab.
Always use `num_workers=0` when a `torch.utils.data.Dataset` reads from Lance.
