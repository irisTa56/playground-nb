# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Jupyter notebook playground for exploring data engineering and ML topics. Notebooks are authored as `.py` files (Jupytext percent format) and synced to `.ipynb` for execution.

## Tooling

- **mise** — task runner (not Makefile). Tools declared in `mise.toml`.
- **uv** — dependency management. Use `uv add --group dev` to add dev deps. Python version is managed by mise.
- **Jupytext** — notebooks are written as `notebooks/**/*.py` (percent format) and synced to `.ipynb` via `mise run nb:sync`.
- **nbstripout** — strips outputs from `.ipynb` before commit.
- **ruff** — linting and formatting. Config uses `extend-select` (not `select`) to preserve defaults.
- **ty** — type checking notebooks with per-script PEP 723 inline metadata.
- **rumdl** — markdown formatting (`rumdl check --fix`).
- **gitleaks** — secret scanning on staged files.
- **lychee** — broken link checking.

## Key Commands

```shell
# Setup
mise install
uv venv --python "$(mise which python)"

# Run all pre-commit checks (secrets, links, markdown, nb sync/strip, lint, typecheck)
mise run pre-commit

# Individual tasks
mise run lint              # ruff check --fix + format
mise run typecheck         # ty check per notebook (uses PEP 723 venvs)
mise run nb:sync           # .py → .ipynb via jupytext
mise run nb:stripout       # strip .ipynb outputs
mise run format-md         # rumdl markdown formatting
mise run secrets           # gitleaks scan
mise run links             # lychee link check

# Set up git pre-commit hook
mise generate git-pre-commit --write --task=pre-commit
```

## Notebook Architecture

- **Source of truth**: `notebooks/<topic>/<name>.py` — Jupytext percent format with `# %%` cell markers.
- **Generated**: `notebooks/<topic>/<name>.ipynb` — built by `mise run nb:sync`, stripped by `mise run nb:stripout`. Do not edit `.ipynb` files directly.
- Each notebook uses **PEP 723 inline script metadata** (`# /// script`) to declare its own dependencies. Type checking runs `uv sync --script` per file to create isolated venvs.
- Notebook `.py` files have `E402` (module-level import order) suppressed in ruff config since cells naturally have interspersed imports.

## Pre-commit

Before finishing a task that modifies files, run `mise run pre-commit` and fix all errors. Do not consider the task complete until pre-commit passes cleanly.

## Conventions

- Notebooks should be self-contained: install their own deps, download data, and detect accelerators.
- The PEP 723 setup cell (`_get_deps`, `_run`, `_setup`) is **identical across all notebooks**. When modifying it, apply the same change to every notebook. Use `/create-notebook` to scaffold new notebooks with the correct boilerplate.
- **Dependency version pins**: When adding or updating PEP 723 dependencies, look up the latest stable release on PyPI and pin the minimum to that minor version (e.g., `torch>=2.11`, not `torch>=2.0`). This keeps notebooks on modern APIs and avoids silent compatibility issues with older versions.
- Use `extend-select` (not `select`) when adding ruff lint rules.
- Python version is pinned in mise.toml `[tools]`. Pass it to uv via `uv venv --python "$(mise which python)"`; do not pin Python separately in uv config.

## Notebook Code Quality

When writing or reviewing notebook code, enforce these rules:

- **No redundant imports.** Don't re-import or alias a symbol already in scope. If an earlier cell imports `Dataset`, later cells must use that name — not `from ... import Dataset as TorchDataset`.
- **Consistent print output across approaches.** When a notebook compares multiple approaches (e.g., Pandas / Polars / Daft), each section must display the **same types of information** in the **same format**. Unify content (row counts, batch shapes, etc.), wording, and prefix style.
- **No redundant content.** Summary/comparison sections at the end should add insight, not restate what earlier sections already showed. If section 2 explains `num_workers`, the final comparison section must not repeat it.
- **No unused variables.** Remove variables that are created but never referenced. In particular, watch for demo datasets or loaders created "for symmetry" but never consumed downstream.
