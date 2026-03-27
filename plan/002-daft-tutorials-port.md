---
goal: Port 4 Daft tutorial notebooks (minhash dedup, embeddings, text-to-image, image color query) into this repo
version: "1.6"
date_created: 2026-03-27
last_updated: 2026-03-27
change_log:
  - date: 2026-03-27
    version: "1.6"
    summary: >
      Phase 7 complete. Split print()+show() in image_color_query.py (3 locations),
      confirmed no issues in other notebooks, added .claude/rules/daft.md with
      show()-separation and progress-bar-suppression rules.
  - date: 2026-03-27
    version: "1.5"
    summary: >
      Phase 2 complete. Implemented MinHash deduplication notebook using
      `daft.datasets.common_crawl()`, built-in `.normalize()`/`.minhash()`,
      LSH banding with self-join, and igraph connected components.
  - date: 2026-03-27
    version: "1.4"
    summary: >
      Phase 5 complete. Moved `import os` out of setup cell in 3 existing
      notebooks, updated CLAUDE.md and `/create-notebook` skill boilerplate.
  - date: 2026-03-27
    version: "1.3"
    summary: >
      Phase 1 complete. Add PAT-009 (source tutorial link), record confirmed
      Daft 0.7 API changes in RISK-004, mark Phase 1 tasks done.
  - date: 2026-03-27
    version: "1.2"
    summary: >
      Bump dependency minimum versions to latest stable releases.
  - date: 2026-03-27
    version: "1.1"
    summary: >
      Address review findings: fix LSH params, specify S3 buckets, correct UDF
      API patterns, add display/testing/error-handling policies.
owner: takayuki
status: In Progress
tags:
  - feature
  - daft
  - notebooks
---

# Introduction

![Status: In Progress](https://img.shields.io/badge/status-In%20Progress-yellow)

Port four Daft tutorial notebooks from [Eventual-Inc/Daft tutorials](https://github.com/Eventual-Inc/Daft/tree/main/tutorials) into this repository, adapting them to the existing notebook style (Jupytext percent format, PEP 723 inline metadata, self-contained setup boilerplate).

Source notebooks:

1. `minhash_dedupe/minhash_dedupe_common_crawl.ipynb` — MinHash deduplication on Common Crawl
2. `embeddings/daft_tutorial_embeddings_stackexchange.ipynb` — Semantic embeddings on StackExchange
3. `text_to_image/text_to_image_generation.ipynb` — Stable Diffusion image generation
4. `image_querying/top_n_red_color.ipynb` — Red-color image querying on OpenImages

## 1. Requirements & Constraints

- **REQ-001**: Each notebook must be a `.py` file in Jupytext percent format under `notebooks/daft/`
- **REQ-002**: PEP 723 setup cell (`_get_deps`, `_run`, `_setup` and the imports they require: `re`, `subprocess`, `sys`, `Path`) must be identical to existing notebooks. `import os` is NOT part of the boilerplate — it belongs in the cell where it is first used (e.g., `os.environ`, `os.cpu_count()`). Source of truth: `notebooks/daft/pytorch_dataloader_tabular.py` lines 25-89 (to be updated by TASK-013).
- **REQ-003**: Each notebook must be self-contained — installs deps, downloads data, detects accelerators
- **REQ-004**: All S3 access must be anonymous (no AWS credentials required)
- **REQ-005**: No `!pip install` or `%pip install` — dependencies handled by PEP 723 + `_setup()`
- **CON-001**: Ray dependency removed from image_color_query — use Daft native runner instead (Ray is ~500MB, has platform issues on macOS ARM)
- **CON-002**: Use stable `daft>=0.7`, not nightly pre-release (text_to_image source used nightly)
- **CON-003**: `E402` (module-level import order) already suppressed in ruff config for notebooks
- **CON-004**: Minimal error handling in UDFs — notebooks are demos, so let exceptions propagate on external resource failures (404, malformed HTML, etc.). No retries or fallbacks needed.
- **GUD-001**: Follow existing naming convention — lowercase with underscores, descriptive
- **GUD-002**: Consistent output formatting across cells (print prefixes, row counts, etc.)
- **GUD-003**: No redundant imports, no unused variables per CLAUDE.md code quality rules
- **PAT-001**: YAML front matter (lines 1-13) identical to existing notebooks
- **PAT-002**: Colab badge markdown cell immediately after front matter
- **PAT-003**: Device detection via `get_device()` helper for **GPU notebooks only** (Phase 3, 4). CPU-only notebooks (Phase 1, 2) must not import `torch` or define `get_device()`.
- **PAT-004**: Version print cell via `importlib.metadata.version()` for key packages
- **PAT-005**: Suppress Daft progress bar via `os.environ["DAFT_PROGRESS_BAR"] = "0"` where appropriate
- **PAT-006**: Image display uses Daft's built-in DataFrame rendering — `df.show()` / `df.collect()` natively renders Image columns inline in Jupyter. No explicit `IPython.display` or `matplotlib.imshow` needed. Exception: text_to_image (Phase 4) uses `IPython.display.display()` + PIL for generated images outside the Daft DataFrame.
- **PAT-007**: `ipywidgets>=8.1` is required for Daft/tqdm Jupyter progress bar rendering. Include in all notebooks.
- **PAT-008**: UDF patterns — follow source tutorials: use `@daft.func()` (element-wise) and `.apply()` (column-wise). Do not use `@daft.udf` (batch/class-based API).
- **PAT-009**: Each notebook's introductory markdown must include a link to the original Daft tutorial it was ported from, with a note that it has been adapted for Daft 0.7+ APIs.

## 2. Implementation Steps

Each phase covers one notebook end-to-end: scaffold, implement content, and pass `mise run pre-commit`. Ordered by complexity (simplest first).

### Phase 1: Image Color Query (`image_color_query.py`)

- GOAL-001: Port `image_querying/top_n_red_color.ipynb` — simplest notebook (no GPU, no heavy models). Validates S3 anonymous access pattern.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create `notebooks/daft/image_color_query.py` with YAML front matter, Colab badge, PEP 723 block (deps: `daft[aws]>=0.7`, `numpy>=2.4`, `Pillow>=12.1`, `ipywidgets>=8.1`), and identical `_get_deps`/`_run`/`_setup` boilerplate. No Ray dependency — use Daft native runner. | ✅ | 2026-03-27 |
| TASK-002 | Implement content — (1) version print (no device detection — CPU only), (2) config constants (`TOP_N=10`, red hue range in HSV), (3) load OpenImages validation images from `s3://daft-public-data/open-images/validation-images/*` with `S3Config(anonymous=True)` (Daft's public mirror — no region override needed), (4) image download via `.download()`, decode via `.decode_image()`, (5) HSV color analysis via `.apply()` + PIL/numpy to compute red-pixel ratio (Daft passes `numpy.ndarray` to `.apply()` — use `Image.fromarray()`), (6) sort by redness score and take top-N, (7) display results via Daft's built-in Image column rendering (`df.show()` / `df.collect()`), (8) summary markdown with source tutorial link (PAT-009) | ✅ | 2026-03-27 |
| TASK-003 | Run `mise run pre-commit` — fix any lint/typecheck/sync issues until clean | ✅ | 2026-03-27 |

### Phase 2: MinHash Deduplication (`minhash_dedup_common_crawl.py`)

- GOAL-002: Port `minhash_dedupe/minhash_dedupe_common_crawl.ipynb` — no GPU but complex pipeline with LSH + graph algorithms.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-004 | Create `notebooks/daft/minhash_dedup_common_crawl.py` with boilerplate and PEP 723 block (deps: `daft[aws]>=0.7`, `selectolax>=0.4`, `scipy>=1.17`, `igraph>=1.0`, `pandas>=3.0`, `matplotlib>=3.10`, `ipywidgets>=8.1`) | ✅ | 2026-03-27 |
| TASK-005 | Implement content — (1) version print cell, (2) config constants (`NUM_WARC_FILES=2`, `K=64`, `NGRAM_SIZE=5`, `LSH_THRESHOLD=0.7`), (3) S3 data ingestion via `daft.datasets.common_crawl()` with `IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))`, (4) HTML parsing via `@daft.func()` + `selectolax`, (5) text normalization via built-in `.normalize()`, (6) MinHash via built-in `.minhash()` (K=64, 5-gram, xxhash), (7) `optimal_param(threshold, K)` helper via scipy to compute `B, R` dynamically with `assert B * R == K`, (8) LSH banding via per-band `.apply()` hash + `union_all`, (9) candidate pair detection via self-join on band hashes, (10) connected components via `igraph`, (11) deduplication — keep one representative per component via left join, (12) matplotlib visualization of duplicate distribution, (13) summary markdown with source tutorial link (PAT-009) | ✅ | 2026-03-27 |
| TASK-006 | Run `mise run pre-commit` — fix any lint/typecheck/sync issues until clean | ✅ | 2026-03-27 |

### Phase 3: Embeddings StackExchange (`embeddings_stackexchange.py`)

- GOAL-003: Port `embeddings/daft_tutorial_embeddings_stackexchange.ipynb` — introduces GPU via SentenceTransformer, medium complexity.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-007 | Create `notebooks/daft/embeddings_stackexchange.py` with boilerplate and PEP 723 block (deps: `daft[aws]>=0.7`, `sentence-transformers>=5.3`, `torch>=2.11`, `accelerate>=1.13`, `ipywidgets>=8.1`) | | |
| TASK-008 | Implement content — (1) version print + `get_device()` detection, (2) config (`MAX_ROWS=10_000`, `MODEL_NAME="all-MiniLM-L6-v2"`), (3) load RedPajama StackExchange parquet from S3 with anonymous access, (4) embedding via `.apply()` with `SentenceTransformer` moved to detected device (model initialized once outside the UDF), (5) compute embeddings on question text, (6) semantic search with `cos_sim()` for a sample query, (7) display top-K results with similarity scores, (8) summary markdown | | |
| TASK-009 | Run `mise run pre-commit` — fix any lint/typecheck/sync issues until clean | | |

### Phase 4: Text-to-Image Generation (`text_to_image_generation.py`)

- GOAL-004: Port `text_to_image/text_to_image_generation.ipynb` — most complex: heavy GPU requirement, large model download, Stable Diffusion.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | Create `notebooks/daft/text_to_image_generation.py` with boilerplate and PEP 723 block (deps: `daft[aws]>=0.7`, `transformers>=5.3`, `diffusers>=0.37`, `torch>=2.11`, `accelerate>=1.13`, `Pillow>=12.1`, `ipywidgets>=8.1`). Use stable `daft>=0.7`, not nightly. | | |
| TASK-011 | Implement content — (1) version print + `get_device()` with explicit CPU warning for Stable Diffusion, (2) config (`NUM_IMAGES=3`, `MODEL_ID="runwayml/stable-diffusion-v1-5"`), (3) load parquet from S3 (url, description, aesthetic_score), filter top-scoring prompts, (4) `StableDiffusionPipeline` setup with dtype per device (`float16` on CUDA, `float32` otherwise), (5) generation via `.apply()` (pipeline is initialized once outside the UDF; do not use `@daft.udf` or `num_gpus`), (6) execute and collect results, (7) display generated images via `IPython.display.display()` + PIL (images are outside Daft DataFrame at this point), (8) summary markdown with model download size warning (~4GB) | | |
| TASK-012 | Run `mise run pre-commit` — fix any lint/typecheck/sync issues until clean | | |

### Phase 5: Boilerplate Cleanup (`import os` separation)

- GOAL-005: Move `import os` out of the PEP 723 setup cell in existing notebooks. `os` is not used by `_get_deps`/`_run`/`_setup` — it belongs in the first cell that needs it. Also update CLAUDE.md and `/create-notebook` skill to reflect the corrected boilerplate boundary.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-013 | In `pytorch_dataloader_tabular.py`, `pytorch_dataloader_text.py`, `pytorch_dataloader_image.py`: move `import os` from the setup cell to the cell where `os` is first used (e.g., `os.environ["DAFT_PROGRESS_BAR"]` or `os.cpu_count()`). Run `mise run pre-commit` to verify. | ✅ | 2026-03-27 |
| TASK-014 | Update CLAUDE.md boilerplate description and `/create-notebook` skill to clarify that the setup cell imports are limited to `re`, `subprocess`, `sys`, `Path`. Notebook-specific stdlib imports like `os` go in the cell where they are first needed. | ✅ | 2026-03-27 |

### Phase 6: ~~Boilerplate Fix~~ (Cancelled)

- ~~GOAL-006~~: Investigation revealed the root cause was NOT `VIRTUAL_ENV` targeting. The `_get_deps()` function's `In[ip.execution_count]` raises `IndexError` in VS Code's Jupyter extension, falling back to `Path(__file__).read_text()`. When another notebook is previewed in VS Code, `__file__` points to that file instead, causing wrong dependencies to be extracted. **Fix**: close other notebook previews before running. No boilerplate change required — the existing fallback mechanism works correctly when `__file__` points to the current file.

### Phase 7: Daft `show()` output fix

- GOAL-007: Daft's `df.show()` overwrites all prior output in the same Jupyter cell. Audit all notebooks for `print()` + `.show()` in the same cell, split into separate cells where needed. Consider adding this as a rule in CLAUDE.md or PAT entry.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-018 | Split `print()` + `df.show()` into separate cells in `minhash_dedup_common_crawl.py` (5 locations). | ✅ | 2026-03-27 |
| TASK-019 | Audit and fix the same issue in existing notebooks (`image_color_query.py`, `pytorch_dataloader_tabular.py`, `pytorch_dataloader_text.py`, `pytorch_dataloader_image.py`, `colpali_vision_retriever.py`). | ✅ | 2026-03-27 |
| TASK-020 | Add a Daft-specific rule to `.claude/rules/` about not mixing `print()` and Daft `.show()` in the same cell (Daft's widget rendering overwrites prior cell output). | ✅ | 2026-03-27 |

## 3. Alternatives

- **ALT-001**: Create sub-topic folders (e.g., `notebooks/daft-dedup/`, `notebooks/daft-embeddings/`) — rejected because the existing convention is flat topic folders by library (`notebooks/daft/`, `notebooks/lance/`).
- **ALT-002**: Keep Ray dependency in image_color_query — rejected because Ray is ~500MB, has macOS ARM issues, and is not required for this notebook's functionality. Daft native runner suffices.
- **ALT-003**: Use Daft nightly pre-release for text_to_image (as in source) — rejected because the repo convention uses stable releases with minimum version pins.
- **ALT-004**: Use `requests` for image downloading in image_color_query — rejected to avoid adding an extra dependency when `urllib.request` from stdlib suffices.

## 4. Dependencies

- **DEP-001**: `daft[aws]>=0.7` — Core DataFrame library with S3 support (all 4 notebooks)
- **DEP-002**: `selectolax>=0.4` — Fast HTML parsing (minhash_dedup)
- **DEP-003**: `scipy>=1.17` — LSH probability integration (minhash_dedup)
- **DEP-004**: `igraph>=1.0` — Graph algorithms for connected components (minhash_dedup)
- **DEP-005**: `sentence-transformers>=5.3` — Semantic embedding models (embeddings)
- **DEP-006**: `diffusers>=0.37` — Stable Diffusion pipeline (text_to_image)
- **DEP-007**: `transformers>=5.3` — Model hub utilities (text_to_image)
- **DEP-008**: `torch>=2.11` — Deep learning framework (embeddings, text_to_image)
- **DEP-009**: `accelerate>=1.13` — Distributed inference support (embeddings, text_to_image)
- **DEP-010**: `Pillow>=12.1` — Image processing (text_to_image, image_color_query)
- **DEP-011**: `numpy>=2.4` — Numerical operations (image_color_query)
- **DEP-012**: `pandas>=3.0`, `matplotlib>=3.10`, `ipywidgets>=8.1` — Visualization/display (minhash_dedup)

## 5. Files

- **FILE-001**: `notebooks/daft/minhash_dedup_common_crawl.py` — New file: MinHash deduplication pipeline
- **FILE-002**: `notebooks/daft/embeddings_stackexchange.py` — New file: Semantic embedding search
- **FILE-003**: `notebooks/daft/text_to_image_generation.py` — New file: Stable Diffusion generation
- **FILE-004**: `notebooks/daft/image_color_query.py` — New file: Red-color image querying
- **FILE-005**: `notebooks/daft/minhash_dedup_common_crawl.ipynb` — Generated by `nb:sync`
- **FILE-006**: `notebooks/daft/embeddings_stackexchange.ipynb` — Generated by `nb:sync`
- **FILE-007**: `notebooks/daft/text_to_image_generation.ipynb` — Generated by `nb:sync`
- **FILE-008**: `notebooks/daft/image_color_query.ipynb` — Generated by `nb:sync`
- **FILE-REF**: `notebooks/daft/pytorch_dataloader_tabular.py` — Reference for boilerplate (lines 25-89)

## 6. Testing

- **TEST-001**: `mise run lint` passes with no errors on all 4 new `.py` files
- **TEST-002**: `mise run typecheck` passes for all 4 new notebooks (PEP 723 venv isolation)
- **TEST-003**: `mise run nb:sync` successfully generates `.ipynb` for all 4 notebooks
- **TEST-004**: `mise run pre-commit` passes end-to-end with no errors
- **TEST-005**: Smoke test for CPU notebooks (Phase 1, 2): `jupyter nbconvert --execute --ExecutePreprocessor.timeout=300` on the generated `.ipynb` to verify cells run without errors. GPU notebooks (Phase 3, 4) cannot be executed in CI — syntax/lint/typecheck coverage (TEST-001 through TEST-004) is sufficient.
- **TEST-006**: Manual verification: open each `.ipynb` in Jupyter/Colab and confirm cells render correctly (markdown, code structure, Colab badge link). For GPU notebooks, verify at minimum that all cells up to the first GPU-dependent cell execute on CPU.

## 7. Risks & Assumptions

- **RISK-001**: S3 anonymous access may require correct `region_name` per bucket — Common Crawl is `us-east-1`. OpenImages data uses Daft's public mirror (`s3://daft-public-data`) which works with default region (no override needed). Mitigation: test each S3 config during implementation.
- **RISK-002**: `selectolax` may lack wheels for some platforms (e.g., older macOS ARM). Mitigation: it has good wheel coverage as of 2025; if issues arise, fall back to `beautifulsoup4`.
- **RISK-003**: Stable Diffusion model download is ~4GB — notebook 4 needs a warning cell about download size and time.
- **RISK-004**: Daft `@daft.func()` and `.apply()` APIs may evolve between releases. Mitigation: pin `>=0.7` and test against current stable. Note: `@daft.udf` (batch/class-based) and `num_gpus` are not used. **Confirmed in Phase 1**: (1) `.url.download()` → `.download()`, `.image.decode()` → `.decode_image()` (accessor namespaces removed), (2) `.apply()` on Image columns passes `numpy.ndarray` (RGB), not `PIL.Image` — use `Image.fromarray()` to convert.
- **RISK-005**: `igraph>=1.0` is a major version bump from the 0.x series — API changes are possible (e.g., function renames). Mitigation: verify `Graph.connected_components()` API against 1.0 docs during implementation. C extension compilation risk remains; pre-built wheels cover most platforms. Verify Colab compatibility during TEST-006.
- **RISK-006**: `daft[aws]` extras pull in `boto3`, which may cause version conflicts with other packages in the PEP 723 venv. Mitigation: PEP 723 venvs are isolated per notebook, so conflicts are limited to each notebook's own dependency set.
- **RISK-007**: `transformers>=5.3` and `sentence-transformers>=5.3` are major version bumps from the source tutorials (which used 4.x and 3.x respectively). Mitigation: verify `StableDiffusionPipeline` and `SentenceTransformer` APIs against current docs during implementation. Pin exact major if breaking changes are found.
- **ASSUMPTION-001**: All S3 data sources (Common Crawl at `us-east-1`, RedPajama, Daft public mirror at `s3://daft-public-data`) remain publicly accessible with anonymous read.
- **ASSUMPTION-002**: The Daft `@daft.func()` and `.apply()` APIs in `>=0.7` support the patterns used in the source tutorials.
- **ASSUMPTION-003**: `runwayml/stable-diffusion-v1-5` remains available on HuggingFace Hub.

## 8. Related Specifications / Further Reading

- [Source tutorials — Eventual-Inc/Daft](https://github.com/Eventual-Inc/Daft/tree/main/tutorials)
- [Existing plan — 001-pytorch-dataloader-tutorial.md](001-pytorch-dataloader-tutorial.md)
- [PEP 723 — Inline script metadata](https://peps.python.org/pep-0723/)
- [Daft documentation](https://www.getdaft.io/projects/docs/en/stable/)
- [Jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format)
