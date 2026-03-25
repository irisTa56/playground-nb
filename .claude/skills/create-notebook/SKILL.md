---
name: create-notebook
description: Scaffold a new self-contained Jupyter notebook (.py percent format) with PEP 723 setup cell, environment verification, device detection, and data download boilerplate
argument-hint: <topic/name> e.g. "daft/pytorch_dataloader_image"
disable-model-invocation: false
---

# Create Notebook

Scaffold a new notebook at `notebooks/$ARGUMENTS.py` in Jupytext percent format.

## Steps

1. Create `notebooks/$ARGUMENTS.py` with the structure defined below. Only the `dependencies` list (Cell 3) and package list (Cell 4) change per notebook. Everything else is verbatim.
2. Run `uv run ruff check --fix` and `uv run ruff format` on the new file.
3. Run `uv run jupytext --to notebook` to generate the `.ipynb`.
4. Run `uv run nbstripout` on the generated `.ipynb`.

## Required notebook structure

### Cell 1: Jupytext header

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: '1.16'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

### Cell 2: Markdown — Title & Colab badge

A `# %% [markdown]` cell with:

- Colab badge: `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/<topic>/<name>.ipynb)`
- Title (e.g. `# PyTorch DataLoader — Tabular Data`)
- Short description of the notebook's topic
- Statement that the notebook is self-contained

### Cell 3: PEP 723 setup cell

**CRITICAL**: The code below (everything after the `# ///` close) is identical across all notebooks. Do NOT modify the implementation — only change the `dependencies` list.

```python
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "<REPLACE with notebook-specific dependencies>",
# ]
# ///

import os
import re
import subprocess
import sys
from pathlib import Path


def _get_deps():
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        src = ip.user_ns.get("In", [""])[ip.execution_count]
    except (NameError, IndexError):
        src = Path(__file__).read_text()

    m = re.search(r"# /// script\s*\n(.*?)# ///", src, re.DOTALL)
    if not m:
        print("Warning: no PEP 723 metadata block found — check the # /// script block")
        return []
    dep_section = re.search(
        r"#\s*dependencies\s*=\s*\[\s*\n(.*?)\n#\s*\]", m.group(1), re.DOTALL
    )
    if not dep_section:
        print("Warning: no dependencies found in PEP 723 block")
        return []
    return re.findall(r'#\s+"([^"]+)"', dep_section.group(1))


def _run(cmd: list[str]) -> None:
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _setup():
    deps = _get_deps()
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        _run([sys.executable, "-m", "pip", "install", "uv"])

    flags = [] if sys.prefix != sys.base_prefix else ["--system"]
    _run(["uv", "pip", "install", *flags, *deps])


_setup()
```

Environment behavior:

| Environment | `_get_deps()` source | `--system` flag |
|---|---|---|
| Colab | `In[n]` from IPython | Yes (no venv) |
| Local Jupyter (uv venv) | `In[n]` from IPython | No (inside venv) |
| `uv run` (.py direct) | `__file__` read | N/A (uv manages) |

### Cell 4: Environment verification

```python
# %%
import importlib.metadata

print(f"Python {sys.version}")
for pkg in ("<REPLACE>", ...):  # list the notebook's key packages
    print(f"  {pkg}: {importlib.metadata.version(pkg)}")
```

### Remaining cells: Notebook content

The actual tutorial content, specific to each notebook.

## Tips

- **Daft progress bar**: Suppress with `os.environ["DAFT_PROGRESS_BAR"] = "0"` before any `daft.read_*()`. Python `logging` does NOT control it.
- **macOS multiprocessing**: Use `num_workers=0` on `sys.platform == "darwin"` for PyTorch DataLoader to avoid fork-related issues.
