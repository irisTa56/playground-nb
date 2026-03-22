---
goal: PyTorch DataLoader Tutorial with Pandas / Polars / Daft comparison
version: "3.0"
date_created: 2026-03-22
last_updated: 2026-03-22
change_log:
  - date: 2026-03-22
    version: "3.0"
    summary: >
      Consolidated 5 notebooks (00_setup, 01_basics, 02_tabular, 03_text, 04_image)
      into 3 self-contained notebooks (pytorch_dataloader_tabular, pytorch_dataloader_text,
      pytorch_dataloader_image). Each notebook includes its own data download, device
      detection, and DataLoader fundamentals relevant to that modality.
      Satisfies REQ-005 strictly.
  - date: 2026-03-22
    version: "2.0"
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

The tutorial is structured as **3 self-contained notebooks**, one per modality (tabular, text, image).
Each notebook independently handles dependency installation, data download, device detection, and introduces DataLoader concepts relevant to that modality.
This means any single notebook can be opened and run without prerequisites.

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
- **REQ-005**: Each notebook must work independently — self-contained setup cell, data download, and device detection
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

### Phase 2 — Tabular Data (`notebooks/pytorch_dataloader_tabular.py`)

- GOAL-002: Self-contained notebook teaching DataLoader fundamentals through a house-price prediction task, comparing Pandas, Polars, and Daft. See [tutorial §1–§2](../docs/pytorch-dataloader-extended.md#1-pytorch-dataloader-fundamentals).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-011 | PEP 723 header cell with `torch`, `pandas`, `polars`, `getdaft`, `scikit-learn`, `matplotlib`, `kaggle` and `_setup()` | ✅ | 2026-03-22 |
| TASK-012 | Environment verification cell: print Python version and key package versions | ✅ | 2026-03-22 |
| TASK-013 | Device detection cell (`get_device()` returning `cuda` / `mps` / `cpu`) | ✅ | 2026-03-22 |
| TASK-014 | Housing dataset download via Kaggle CLI with existence check and manual download URL fallback | ✅ | 2026-03-22 |
| TASK-015 | Explain `Dataset` ↔ `DataLoader` relationship in Markdown cells | ✅ | 2026-03-22 |
| TASK-016 | Minimal `TensorDataset` + `DataLoader` example as warm-up | ✅ | 2026-03-22 |
| TASK-017 | Walk through core parameters: `batch_size`, `shuffle`, `num_workers`, `pin_memory`, `collate_fn` (overview only; deep-dive in text notebook) | ✅ | 2026-03-22 |
| TASK-018 | macOS note: `multiprocessing_context="spawn"` helper (`make_dataloader`) | ✅ | 2026-03-22 |
| TASK-019 | Pandas preprocessing (§2.1): `read_csv` → encoding → `StandardScaler` → `train_test_split` → `.to_numpy()` | ✅ | 2026-03-22 |
| TASK-020 | Polars preprocessing (§2.2): `read_csv` → expression-based encoding → `.to_numpy()` (Arrow zero-copy) | ✅ | 2026-03-22 |
| TASK-021 | Polars Lazy API (§2.2): `scan_csv` → predicate pushdown / projection pushdown demo | ✅ | 2026-03-22 |
| TASK-022 | Daft preprocessing (§2.3): `read_csv` → UDF → `.collect()` → tensor conversion | ✅ | 2026-03-22 |
| TASK-023 | Shared `HousePriceDataset(Dataset)` class (§2.4) | ✅ | 2026-03-22 |
| TASK-024 | Lightweight training loop (few epochs, `nn.Linear`) to verify end-to-end | ✅ | 2026-03-22 |
| TASK-025 | Comparison table Markdown cell (§2.5) | ✅ | 2026-03-22 |
| TASK-026 | Generate `.ipynb` and validate | ✅ | 2026-03-22 |

### Phase 3 — Text Data (`notebooks/pytorch_dataloader_text.py`)

- GOAL-003: Self-contained notebook for IMDB sentiment analysis with BERT tokenization, comparing CSV loading across libraries. See [tutorial §3](../docs/pytorch-dataloader-extended.md#3-text-data-bert--imdb-sentiment-classification).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-027 | PEP 723 header cell with `torch`, `pandas`, `polars`, `getdaft`, `transformers`, `kaggle` and `_setup()` | | |
| TASK-028 | Environment verification cell: print Python version and key package versions | | |
| TASK-029 | Device detection cell (`get_device()`) | | |
| TASK-030 | IMDB dataset download via Kaggle CLI with existence check and manual download URL fallback | | |
| TASK-031 | Introduce `collate_fn` for variable-length sequences — the DataLoader concept most relevant to text | | |
| TASK-032 | CSV loading comparison (§3.1): `%%time` across Pandas, Polars, Daft | | |
| TASK-033 | Tokenisation example (§3.2): `BertTokenizer.from_pretrained("bert-base-uncased")` | | |
| TASK-034 | `IMDBDataset(Dataset)` class (§3.2) returning `input_ids`, `attention_mask`, `label` | | |
| TASK-035 | DataLoader config: `batch_size=16`, `num_workers`, `pin_memory` | | |
| TASK-036 | Batch verification: print tensor shapes | | |
| TASK-037 | Generate `.ipynb` and validate | | |

### Phase 4 — Image Data (`notebooks/pytorch_dataloader_image.py`)

- GOAL-004: Self-contained notebook for Cat/Dog classification comparing torchvision, Polars metadata, and Daft multimodal pipelines. See [tutorial §4](../docs/pytorch-dataloader-extended.md#4-image-data-where-library-differences-are-most-pronounced).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-038 | PEP 723 header cell with `torch`, `torchvision`, `polars`, `getdaft`, `Pillow`, `matplotlib`, `kaggle` and `_setup()` | | |
| TASK-039 | Environment verification cell: print Python version and key package versions | | |
| TASK-040 | Device detection cell (`get_device()`) | | |
| TASK-041 | Cat/Dog dataset download via Kaggle CLI with existence check and manual download URL fallback | | |
| TASK-042 | Introduce `num_workers` and `pin_memory` tuning — the DataLoader concepts most relevant to image I/O | | |
| TASK-043 | torchvision approach (§4.1): `transforms.v2.Compose` → `ImageFolder` → `DataLoader` | | |
| TASK-044 | Batch visualisation with matplotlib | | |
| TASK-045 | Polars metadata management (§4.2): scan dirs → DataFrame of paths + labels → custom `Dataset` | | |
| TASK-046 | Daft multimodal pipeline (§4.3): `from_glob_path` → label extraction → `url.download()` → `decode_image()` | | |
| TASK-047 | Daft → `IterableDataset` → `DataLoader` (§4.4) | | |
| TASK-048 | Comparison table Markdown cell (§4.5) | | |
| TASK-049 | Generate `.ipynb` and validate | | |

### Phase 5 — Finalisation

- GOAL-005: Documentation, full validation, and Colab smoke test.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-050 | Create `projects/daft/README.md` with setup instructions (local + Colab) | | |
| TASK-051 | Run `mise run nb:sync` to regenerate all `.ipynb` from `.py` | | |
| TASK-052 | Run `mise run lint` (`ruff check --fix` + `ruff format`) on all `.py` notebooks and confirm no violations | | |
| TASK-053 | Run `mise run typecheck` (`ty check`) on all `.py` notebooks and resolve any type errors | | |
| TASK-054 | Run all notebooks top-to-bottom locally and confirm no errors | | |
| TASK-055 | Open each `.ipynb` on Colab via GitHub and confirm the `_setup()` cell installs deps correctly | | |

## 3. Alternatives

- **ALT-001**: Manage all runtime dependencies in `pyproject.toml` — rejected because Colab cannot read `pyproject.toml`; PEP 723 inline metadata keeps each notebook self-contained and works across all environments.
- **ALT-002**: Use only `.ipynb` for git — rejected because `.ipynb` JSON diffs are noisy; `.py` percent format gives clean line-based diffs while `.ipynb` is co-committed (output-stripped) for Colab.
- **ALT-003**: Use `requirements.txt` per notebook — rejected; PEP 723 block is a single source of truth embedded directly in the code with no extra files to maintain.
- **ALT-004**: Use `pip` instead of `uv` — rejected because `uv` is significantly faster and the `_setup()` pattern auto-installs `uv` via pip as a fallback when it is not available (e.g., on Colab).
- **ALT-005**: Use `Makefile` for task automation — rejected in favour of `mise` tasks; `mise.toml` already manages tool versions, adding tasks keeps all project automation in one place with dependency ordering (`depends`, `wait_for`) and avoids requiring `make` on all platforms.
- **ALT-006**: Separate setup notebook (`00_setup`) and basics notebook (`01_dataloader_basics`) — rejected because it violates REQ-005 (each notebook must work independently). Data download and DataLoader fundamentals are instead embedded into each modality notebook so any single notebook can be opened and run without prerequisites.

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
    ├── pytorch_dataloader_tabular.py      # .py percent format (source of truth)
    ├── pytorch_dataloader_tabular.ipynb   # Generated; output-stripped
    ├── pytorch_dataloader_text.py
    ├── pytorch_dataloader_text.ipynb
    ├── pytorch_dataloader_image.py
    └── pytorch_dataloader_image.ipynb
```

- **FILE-001**: `pyproject.toml` — project metadata + dev dependencies only
- **FILE-002**: `mise.toml` — task runner config (`nb:sync`, `nb:clean`, `lint`, `typecheck`, `pre-commit`)
- **FILE-003**: `.gitattributes` — `nbstripout` filter registration
- **FILE-004**: `notebooks/pytorch_dataloader_tabular.py` — Tabular: DataLoader fundamentals + Pandas vs Polars vs Daft
- **FILE-005**: `notebooks/pytorch_dataloader_text.py` — Text: IMDB + BERT + collate_fn
- **FILE-006**: `notebooks/pytorch_dataloader_image.py` — Image: Cat/Dog + num_workers/pin_memory tuning

## 6. Testing

- **TEST-001**: `uv sync` completes without errors
- **TEST-002**: Each notebook's `_setup()` cell installs all its dependencies from scratch
- **TEST-003**: Each notebook downloads its dataset via Kaggle CLI when `data/` is empty
- **TEST-004**: `mise run nb:sync` generates valid `.ipynb` for every `.py`
- **TEST-005**: `nbstripout` correctly strips outputs from committed `.ipynb` files
- **TEST-006**: `ruff check` reports no violations on all `.py` notebooks
- **TEST-007**: `ruff format --check` reports no formatting differences on all `.py` notebooks
- **TEST-008**: `ty check` reports no type errors on all `.py` notebooks
- **TEST-009**: Each notebook runs top-to-bottom locally without errors (independently, in any order)
- **TEST-010**: Each `DataLoader` yields batches with expected tensor shapes
- **TEST-011**: `_setup()` cell installs dependencies correctly on Colab (no pre-existing venv)
- **TEST-012**: `_setup()` cell works in local venv (no `--system` flag)
- **TEST-013**: VS Code detects `.venv` kernel and can run `# %%` cells from `.py` files

## 7. Risks & Assumptions

- **RISK-001**: Daft API may change between versions — pin minimum version and note in notebooks to check latest docs
- **RISK-002**: `nbstripout` filter must be installed per-clone; if missing, outputs will leak into diffs
- **RISK-003**: PEP 723 parsing in `_setup()` relies on regex; malformed metadata blocks will silently fail — mitigate by printing a warning when no dependencies are found (`if not deps: print("Warning: no PEP 723 dependencies found — check the # /// script block")`)
- **RISK-004**: `_setup()`, `get_device()`, and Kaggle download logic are duplicated across all 3 notebooks to satisfy REQ-005. Changes to the shared pattern must be applied to every notebook — mitigate by keeping the boilerplate minimal and identical so a simple search-and-replace suffices
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
