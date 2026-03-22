---
goal: PyTorch DataLoader Tutorial with Pandas / Polars / Daft comparison
version: "2.0"
date_created: 2026-03-22
last_updated: 2026-03-22
change_log:
  - date: 2026-03-22
    commits: [6ce1c52, 731404e]
    summary: >
      Phase 1 tooling setup largely completed. Replaced Makefile with mise tasks.
      Root pre-commit now delegates to projects/daft/ when staged files are detected.
      Python 3.12 pinned via .python-version (Colab compatible); requires-python >=3.12.
      All tool invocations use `uv run` prefix. ty config updated to [tool.ty.environment].
status: "In progress"
tags:
  - feature
  - tutorial
  - pytorch
  - notebook
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)

Implement the content of [`pytorch-dataloader-extended.md`](../docs/pytorch-dataloader-extended.md) step-by-step under `projects/daft/`, starting from environment setup.
The goal is to learn how to use PyTorch DataLoader across tabular, text, and image modalities while comparing three libraries: Pandas, Polars, and Daft.

Notebooks are authored as **`.py` (percent format)** and kept as the single source of truth.
`.ipynb` files are generated via `jupytext` and co-committed for Colab compatibility.
Dependencies are declared inline using PEP 723 script metadata and installed at runtime via `uv pip install`.

Code samples for each notebook are defined in the [extended tutorial doc](../docs/pytorch-dataloader-extended.md).
This plan focuses on the implementation structure, notebook tooling, and task tracking.

## 1. Requirements & Constraints

- **REQ-001**: `.py` (percent format) files are the canonical source; `.ipynb` is derived
- **REQ-002**: `.ipynb` must be co-committed so Colab's "Open from GitHub" works
- **REQ-003**: Dependencies are declared once — in a PEP 723 `# /// script` block at the top of each notebook
- **REQ-004**: The same notebook must run on Colab, local Jupyter (venv), and via `uv run` without modification
- **REQ-005**: Each notebook must work independently (self-contained setup cell)
- **SEC-001**: Kaggle API token (`~/.kaggle/kaggle.json`) must never be committed
- **CON-001**: Apple Silicon (MPS) must be supported; `multiprocessing_context="spawn"` required when `num_workers > 0` on macOS
- **CON-002**: `bert-base-uncased` is ~440 MB; first download latency must be noted
- **CON-003**: Daft API is evolving rapidly; pin minimum version and note to check latest docs
- **GUD-001**: Use `nbstripout` via `.gitattributes` to keep `.ipynb` diffs clean
- **GUD-002**: Use `mise` tasks (`nb:sync`, `lint`, `typecheck`, `pre-commit`) for automation instead of `Makefile`
- **GUD-003**: Use `ruff` for linting and formatting `.py` notebooks; use `ty` for type checking
- **GUD-004**: Root `mise.toml` pre-commit hook delegates to `projects/daft/mise.toml` when staged files are in `projects/daft/`
- **GUD-005**: All tool invocations (`jupytext`, `ruff`, `ty`) use `uv run` prefix to ensure correct environment
- **PAT-001**: PEP 723 inline metadata + runtime `_setup()` pattern (see Appendix)

## 2. Implementation Steps

### Phase 1 — Repository & Tooling Setup

- GOAL-001: Configure the project directory, dependency tooling, and git hooks so that `.py` notebooks are the source of truth and `.ipynb` is auto-generated and output-stripped.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create `projects/daft/mise.toml` with `uv = "latest"` and `.python-version` pinning Python 3.12 (Colab compatible); Python is managed by `uv`, not `mise` | ✅ | 2026-03-22 |
| TASK-002 | Create `projects/daft/.gitignore` excluding `data/`, `.venv/`, `__pycache__/` | ✅ | 2026-03-22 |
| TASK-003 | Create `projects/daft/.gitattributes` with `*.ipynb filter=nbstripout` | ✅ | 2026-03-22 |
| TASK-004 | Create mise tasks in `projects/daft/mise.toml`: `nb:sync` (jupytext conversion), `nb:clean`, `lint`, `typecheck`, `pre-commit` (depends on all three) | ✅ | 2026-03-22 |
| TASK-005 | Create `projects/daft/pyproject.toml` — minimal project metadata (`requires-python = ">=3.12"`) with dev dependencies only (`jupytext`, `nbstripout`, `ipykernel`, `ruff`, `ty`); runtime deps live in each notebook's PEP 723 block | ✅ | 2026-03-22 |
| TASK-006 | Configure `ruff` in `pyproject.toml`: `target-version = "py312"`, select rules (`E`, `F`, `I`, `UP`), and `[tool.ruff.lint.per-file-ignores]` with `"notebooks/*.py" = ["E402"]` to allow late imports after the `_setup()` cell | ✅ | 2026-03-22 |
| TASK-007 | Configure `ty` in `pyproject.toml`: set `[tool.ty.environment]` with `root = ["notebooks/"]` | ✅ | 2026-03-22 |
| TASK-008 | Add root `mise.toml` pre-commit hook that delegates to `projects/daft/mise run pre-commit` when staged files are in `projects/daft/` | ✅ | 2026-03-22 |
| TASK-009 | Run `uv sync` and set up `.venv` with `ipykernel` for local VS Code kernel | ✅ | 2026-03-22 |
| TASK-010 | Verify VS Code detects `.venv` kernel and can run `# %%` cells natively from `.py` files | | |

### Phase 2 — Setup Notebook (`notebooks/00_setup.py`)

- GOAL-002: Create the environment-check and data-download notebook with the PEP 723 setup cell pattern.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-011 | Write PEP 723 header cell with all runtime dependencies and `_setup()` bootstrap function | | |
| TASK-012 | Environment verification cell: print Python version and all package versions | | |
| TASK-013 | Device detection cell (`get_device()` returning `cuda` / `mps` / `cpu`) | | |
| TASK-014 | Dataset download cells via Kaggle CLI with existence checks (Housing, IMDB, Cat/Dog) | | |
| TASK-015 | Directory structure verification cell after download | | |
| TASK-016 | Generate `.ipynb` with `jupytext --to notebook notebooks/00_setup.py` and verify on Colab | | |

### Phase 3 — DataLoader Fundamentals (`notebooks/01_dataloader_basics.py`)

- GOAL-003: Teach `Dataset` / `DataLoader` concepts with minimal examples. See [tutorial §1](../docs/pytorch-dataloader-extended.md#1-pytorch-dataloader-fundamentals).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-017 | PEP 723 header cell with `torch` dependency and `_setup()` | | |
| TASK-018 | Explain `Dataset` ↔ `DataLoader` relationship in Markdown cells | | |
| TASK-019 | Minimal `TensorDataset` + `DataLoader` example | | |
| TASK-020 | Walk through parameters: `batch_size`, `shuffle`, `num_workers`, `pin_memory`, `collate_fn` | | |
| TASK-021 | macOS note: `multiprocessing_context="spawn"` helper (`make_dataloader`) | | |
| TASK-022 | Generate `.ipynb` and validate | | |

### Phase 4 — Tabular Data (`notebooks/02_tabular.py`)

- GOAL-004: Compare Pandas, Polars, and Daft for tabular preprocessing → DataLoader pipeline. See [tutorial §2](../docs/pytorch-dataloader-extended.md#2-tabular-data-where-the-choice-of-preprocessing-library-matters-most).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-023 | PEP 723 header cell with `torch`, `pandas`, `polars`, `getdaft`, `scikit-learn`, `matplotlib` | | |
| TASK-024 | Pandas preprocessing (§2.1): `read_csv` → encoding → `StandardScaler` → `train_test_split` → `.to_numpy()` | | |
| TASK-025 | Polars preprocessing (§2.2): `read_csv` → expression-based encoding → `.to_numpy()` (Arrow zero-copy) | | |
| TASK-026 | Polars Lazy API (§2.2): `scan_csv` → predicate pushdown / projection pushdown demo | | |
| TASK-027 | Daft preprocessing (§2.3): `read_csv` → UDF → `.collect()` → tensor conversion | | |
| TASK-028 | Shared `HousePriceDataset(Dataset)` class (§2.4) | | |
| TASK-029 | Lightweight training loop (few epochs, `nn.Linear`) to verify end-to-end | | |
| TASK-030 | Comparison table Markdown cell (§2.5) | | |
| TASK-031 | Generate `.ipynb` and validate | | |

### Phase 5 — Text Data (`notebooks/03_text.py`)

- GOAL-005: IMDB sentiment analysis with BERT tokenization, comparing CSV loading across libraries. See [tutorial §3](../docs/pytorch-dataloader-extended.md#3-text-data-bert--imdb-sentiment-classification).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-032 | PEP 723 header cell with `torch`, `pandas`, `polars`, `getdaft`, `transformers` | | |
| TASK-033 | CSV loading comparison (§3.1): `%%time` across Pandas, Polars, Daft | | |
| TASK-034 | Tokenisation example (§3.2): `BertTokenizer.from_pretrained("bert-base-uncased")` | | |
| TASK-035 | `IMDBDataset(Dataset)` class (§3.2) returning `input_ids`, `attention_mask`, `label` | | |
| TASK-036 | DataLoader config: `batch_size=16`, `num_workers`, `pin_memory` | | |
| TASK-037 | Batch verification: print tensor shapes | | |
| TASK-038 | Generate `.ipynb` and validate | | |

### Phase 6 — Image Data (`notebooks/04_image.py`)

- GOAL-006: Cat/Dog classification comparing torchvision, Polars metadata, and Daft multimodal pipelines. See [tutorial §4](../docs/pytorch-dataloader-extended.md#4-image-data-where-library-differences-are-most-pronounced).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-039 | PEP 723 header cell with `torch`, `torchvision`, `polars`, `getdaft`, `Pillow`, `matplotlib` | | |
| TASK-040 | torchvision approach (§4.1): `transforms.v2.Compose` → `ImageFolder` → `DataLoader` | | |
| TASK-041 | Batch visualisation with matplotlib | | |
| TASK-042 | Polars metadata management (§4.2): scan dirs → DataFrame of paths + labels → custom `Dataset` | | |
| TASK-043 | Daft multimodal pipeline (§4.3): `from_glob_path` → label extraction → `url.download()` → `decode_image()` | | |
| TASK-044 | Daft → `IterableDataset` → `DataLoader` (§4.4) | | |
| TASK-045 | Comparison table Markdown cell (§4.5) | | |
| TASK-046 | Generate `.ipynb` and validate | | |

### Phase 7 — Finalisation

- GOAL-007: Documentation, full validation, and Colab smoke test.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-047 | Create `projects/daft/README.md` with setup instructions (local + Colab) | | |
| TASK-048 | Run `mise run nb:sync` to regenerate all `.ipynb` from `.py` | | |
| TASK-049 | Run `mise run lint` (`ruff check --fix` + `ruff format`) on all `.py` notebooks and confirm no violations | | |
| TASK-050 | Run `mise run typecheck` (`ty check`) on all `.py` notebooks and resolve any type errors | | |
| TASK-051 | Run all notebooks top-to-bottom locally and confirm no errors | | |
| TASK-052 | Open each `.ipynb` on Colab via GitHub and confirm the `_setup()` cell installs deps correctly | | |

## 3. Alternatives

- **ALT-001**: Manage all runtime dependencies in `pyproject.toml` — rejected because Colab cannot read `pyproject.toml`; PEP 723 inline metadata keeps each notebook self-contained and works across all environments.
- **ALT-002**: Use only `.ipynb` for git — rejected because `.ipynb` JSON diffs are noisy; `.py` percent format gives clean line-based diffs while `.ipynb` is co-committed (output-stripped) for Colab.
- **ALT-003**: Use `requirements.txt` per notebook — rejected; PEP 723 block is a single source of truth embedded directly in the code with no extra files to maintain.
- **ALT-004**: Use `pip` instead of `uv` — rejected because `uv` is significantly faster and the `_setup()` pattern auto-installs `uv` via pip as a fallback when it is not available (e.g., on Colab).
- **ALT-005**: Use `Makefile` for task automation — rejected in favour of `mise` tasks; `mise.toml` already manages tool versions, adding tasks keeps all project automation in one place with dependency ordering (`depends`, `wait_for`) and avoids requiring `make` on all platforms.

## 4. Dependencies

### Dev dependencies (in `pyproject.toml`)

- **DEP-001**: `jupytext>=1.16` — `.py` ↔ `.ipynb` conversion
- **DEP-002**: `nbstripout>=0.7` — strip cell outputs from `.ipynb` before commit
- **DEP-003**: `ipykernel>=6.29` — register the venv as a Jupyter kernel for VS Code
- **DEP-004**: `ruff>=0.11` — linting and formatting for `.py` notebooks
- **DEP-005**: `ty>=0.0.1a7` — type checking for `.py` notebooks

### Runtime dependencies (in each notebook's PEP 723 block)

- **DEP-006**: `uv` — fast package installer; auto-installed by `_setup()` if missing
- **DEP-007**: `torch>=2.2`, `torchvision>=0.17` — core PyTorch
- **DEP-008**: `pandas>=2.2`, `polars>=1.0`, `getdaft>=0.4` — DataFrame libraries to compare
- **DEP-009**: `transformers>=4.40` — BERT tokenizer for text notebook
- **DEP-010**: `scikit-learn>=1.4` — preprocessing utilities
- **DEP-011**: `Pillow>=10.0`, `matplotlib>=3.8` — image handling and plotting
- **DEP-012**: `kaggle>=1.6` — dataset download CLI

## 5. Files

```text
projects/daft/
├── pyproject.toml              # Dev-only deps (jupytext, nbstripout, ipykernel)
├── uv.lock
├── mise.toml                   # Pins uv; defines nb:sync, lint, typecheck, pre-commit tasks
├── .python-version             # Pins Python 3.12 (Colab compatible; managed by uv)
├── .gitignore                  # Excludes data/, .venv/, __pycache__/
├── .gitattributes              # *.ipynb filter=nbstripout
├── README.md                   # Setup instructions (local + Colab)
├── plan/
│   └── 001-pytorch-dataloader-tutorial.md
├── docs/
│   └── pytorch-dataloader-extended.md
├── data/                       # Gitignored
│   ├── housing/
│   ├── imdb/
│   └── cats_and_dogs/
└── notebooks/
    ├── 00_setup.py             # .py percent format (source of truth)
    ├── 00_setup.ipynb          # Generated; output-stripped
    ├── 01_dataloader_basics.py
    ├── 01_dataloader_basics.ipynb
    ├── 02_tabular.py
    ├── 02_tabular.ipynb
    ├── 03_text.py
    ├── 03_text.ipynb
    ├── 04_image.py
    └── 04_image.ipynb
```

- **FILE-001**: `pyproject.toml` — project metadata + dev dependencies only
- **FILE-002**: `mise.toml` — task runner config (`nb:sync`, `nb:clean`, `lint`, `typecheck`, `pre-commit`)
- **FILE-003**: `.gitattributes` — `nbstripout` filter registration
- **FILE-004**: `notebooks/00_setup.py` — Environment check + data download
- **FILE-005**: `notebooks/01_dataloader_basics.py` — DataLoader fundamentals
- **FILE-006**: `notebooks/02_tabular.py` — Tabular: Pandas vs Polars vs Daft
- **FILE-007**: `notebooks/03_text.py` — Text: IMDB + BERT
- **FILE-008**: `notebooks/04_image.py` — Image: Cat/Dog classification

## 6. Testing

- **TEST-001**: `uv sync` completes without errors
- **TEST-002**: All packages can be imported in `00_setup` notebook
- **TEST-003**: Kaggle datasets are correctly extracted under `data/`
- **TEST-004**: `mise run nb:sync` generates valid `.ipynb` for every `.py`
- **TEST-005**: `nbstripout` correctly strips outputs from committed `.ipynb` files
- **TEST-006**: `ruff check` reports no violations on all `.py` notebooks
- **TEST-007**: `ruff format --check` reports no formatting differences on all `.py` notebooks
- **TEST-008**: `ty check` reports no type errors on all `.py` notebooks
- **TEST-009**: Notebooks `01` through `04` run top-to-bottom locally without errors
- **TEST-010**: Each `DataLoader` yields batches with expected tensor shapes
- **TEST-011**: `_setup()` cell installs dependencies correctly on Colab (no pre-existing venv)
- **TEST-012**: `_setup()` cell works in local venv (no `--system` flag)
- **TEST-013**: VS Code detects `.venv` kernel and can run `# %%` cells from `.py` files

## 7. Risks & Assumptions

- **RISK-001**: Daft API may change between versions — pin minimum version and note in notebooks to check latest docs
- **RISK-002**: `nbstripout` filter must be installed per-clone; if missing, outputs will leak into diffs
- **RISK-003**: PEP 723 parsing in `_setup()` relies on regex; malformed metadata blocks will silently fail — mitigate by printing a warning when no dependencies are found (`if not deps: print("Warning: no PEP 723 dependencies found — check the # /// script block")`)
- **ASSUMPTION-001**: Users have a Kaggle account and `~/.kaggle/kaggle.json` configured. Notebooks should provide a manual download URL as a fallback for users without Kaggle API access
- **ASSUMPTION-002**: `uv` is either pre-installed or can be installed via `pip install uv` (Colab fallback)
- **ASSUMPTION-003**: Colab provides `get_ipython()` and `In[n]` in the user namespace
- **ASSUMPTION-004**: VS Code with Python + Jupyter extensions is used for local development

## 8. Related Specifications / Further Reading

- [`pytorch-dataloader-extended.md`](../docs/pytorch-dataloader-extended.md) — full tutorial with all code samples
- [PEP 723 — Inline script metadata](https://peps.python.org/pep-0723/)
- [Jupytext documentation](https://jupytext.readthedocs.io/en/latest/)
- [nbstripout](https://github.com/kynan/nbstripout)
- [uv documentation](https://docs.astral.sh/uv/)

## Appendix: PEP 723 Setup Cell Pattern

Each notebook starts with the following cell.
It declares dependencies inline and installs them at runtime via `uv pip install`, auto-installing `uv` itself if needed.

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.2",
#   "polars>=1.0",
#   ...
# ]
# ///

import re, sys, subprocess

def _get_deps():
    try:
        ip = get_ipython()
        src = ip.user_ns.get("In", [])[ip.execution_count]
    except NameError:
        src = open(__file__).read()

    m = re.search(r"# /// script\s*\n(.*?)# ///", src, re.DOTALL)
    if not m:
        print("Warning: no PEP 723 metadata block found — check the # /// script block")
        return []
    # Extract only the "dependencies" list, ignoring other fields like requires-python
    dep_section = re.search(r"#\s*dependencies\s*=\s*\[\s*\n(.*?)\n#\s*\]", m.group(1), re.DOTALL)
    if not dep_section:
        print("Warning: no dependencies found in PEP 723 block")
        return []
    return re.findall(r'#\s+"([^"]+)"', dep_section.group(1))

def _setup():
    deps = _get_deps()
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv", "-q"], check=True)

    flags = [] if sys.prefix != sys.base_prefix else ["--system"]
    subprocess.run(["uv", "pip", "install", "-q"] + flags + deps, check=True)

_setup()
```

### Environment Behavior

| Environment | `_get_deps()` source | `--system` flag |
|---|---|---|
| Colab | `In[n]` from IPython | Yes (no venv) |
| Local Jupyter (uv venv) | `In[n]` from IPython | No (inside venv) |
| Local Jupyter (system Python) | `In[n]` from IPython | Yes (no venv) |
| `uv run` (`.py` direct execution) | `__file__` read | N/A (uv manages) |

### Notes

- The setup cell **must be the first cell** in every notebook (`In[n]` index depends on it)
- After kernel restart, packages requiring reload (e.g., numpy major version swap) need re-execution
- VS Code local development: `# %%` cell markers in `.py` files are run natively — **`.ipynb` is not needed**
