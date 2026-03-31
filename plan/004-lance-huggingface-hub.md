---
goal: Add a Lance x Hugging Face Hub notebook demonstrating remote dataset access, blob handling, and vector search
version: "1.1"
date_created: 2026-03-31
last_updated: 2026-03-31
change_log:
  - date: 2026-03-31
    version: "1.1"
    summary: >
      Review fixes: updated version pins (pylance>=4.0, lancedb>=0.30,
      torchcodec>=0.11), added matplotlib to TASK-002, replaced dummy
      vector with existing-row embedding approach, expanded FineWeb-Edu
      note in ALT-001, added hf:// ASSUMPTION, removed torchcodec
      fallback (macOS ARM confirmed supported), added network test
      ASSUMPTION.
  - date: 2026-03-31
    version: "1.0"
    summary: >
      Initial plan based on https://lancedb.com/blog/lance-x-huggingface-a-new-era-of-sharing-multimodal-data/
owner: takayuki
status: Planned
tags:
  - feature
  - notebook
  - lance
  - lancedb
  - huggingface
  - multimodal
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

Add a self-contained notebook `notebooks/lance/lance_huggingface_hub.py` that demonstrates the Lance format integration with Hugging Face Hub, based on the [Lance x Hugging Face blog post](https://lancedb.com/blog/lance-x-huggingface-a-new-era-of-sharing-multimodal-data/).
The notebook walks through three capabilities announced in the blog:

1. **Remote dataset access** — open Lance datasets hosted on Hugging Face Hub via `hf://` URIs, scan metadata with column selection and filters without downloading full blobs.
2. **Blob handling** — fetch large binary objects (video) on demand using Lance's blob API, decode with `torchcodec`.
3. **Vector search** — run nearest-neighbor search on bundled embeddings via LanceDB, directly from the Hub.

This complements the existing `clip_multimodal_lance.py` (local Lance workflow) and `colpali_vision_retriever.py` (LanceDB retrieval) by showcasing **remote-first** access patterns.

## 1. Requirements & Constraints

- **REQ-001**: Demonstrate all three key features from the blog: remote metadata scanning with filters, blob access for video, and vector search via LanceDB.
- **REQ-002**: Use the public Hub datasets referenced in the blog (`lance-format/Openvid-1M`, `lance-format/laion-1m`) — no authentication required.
- **REQ-003**: Use `hf://datasets/...` URIs for all remote dataset access (both `lance.dataset()` and `lancedb.connect()`).
- **SEC-001**: No hardcoded credentials or API keys. All datasets must be publicly accessible.
- **CON-001**: Network access is required — the notebook operates on remote datasets, not local copies. Add a note about this in the introduction.
- **CON-002**: Video blob downloads can be large. Limit demo to 1-2 blobs and document expected sizes.
- **CON-003**: Vector search query embedding dimensions must match the dataset's embedding column (768-dim for LAION `img_emb`). Use an existing row's embedding as the query vector to produce meaningful results without loading a separate model.
- **GUD-001**: Each section should print enough output to verify correctness (row counts, schema, sample metadata, search results) without being verbose.
- **GUD-002**: Explicitly contrast remote vs local access patterns in markdown cells to highlight the value proposition.
- **PAT-001**: Follow the same boilerplate (`_get_deps`, `_run`, `_setup`) as existing notebooks.
- **PAT-002**: Use `num_workers=0` if any DataLoader reads from Lance (per `.claude/rules/lance.md`).
- **PAT-003**: PyPI package for Lance is `pylance` (not `lance`). LanceDB is `lancedb`.

## 2. Implementation Steps

### Phase 1: Notebook Scaffolding

- GOAL-001: Create the notebook file with correct boilerplate, PEP 723 metadata, and introductory markdown.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Scaffold `notebooks/lance/lance_huggingface_hub.py` using `/create-notebook` skill. Add Colab badge, title, and description markdown linking to the blog post. Note that network access is required | | |
| TASK-002 | Add PEP 723 `# /// script` block with dependencies: `pylance>=4.0`, `lancedb>=0.30`, `pyarrow>=23.0`, `torchcodec>=0.11`, `torch>=2.11`, `matplotlib>=3.10`, `Pillow>=12.1`, `numpy>=2.4`. Pin versions to latest stable at implementation time. Look up actual latest versions on PyPI before writing | | |
| TASK-003 | Verify `_get_deps`, `_run`, `_setup` boilerplate matches existing notebooks exactly | | |
| TASK-004 | Add environment/version verification cell printing installed package versions | | |

### Phase 2: Remote Dataset Access — Metadata Scanning & Filtering

- GOAL-002: Demonstrate opening a Lance dataset from HF Hub and scanning metadata efficiently without downloading blobs.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Add markdown cell explaining Lance's remote dataset access via `hf://` URIs, how range reads enable efficient metadata-only scanning, and how blobs are skipped unless explicitly requested | | |
| TASK-006 | Open the OpenVid-1M dataset via `lance.dataset("hf://datasets/lance-format/Openvid-1M/data/train.lance")`. Print schema, row count, and column names | | |
| TASK-007 | Demonstrate column-selective scanning: use `ds.scanner(columns=["caption", "aesthetic_score"], filter="aesthetic_score >= 4.5", limit=10)` to fetch metadata without loading video blobs. Print results as a list of dicts | | |
| TASK-008 | Add a note comparing this to the traditional workflow (download entire dataset, then filter locally) to highlight I/O savings | | |

### Phase 3: Blob Handling — Video Access & Decoding

- GOAL-003: Fetch video blobs on demand and decode them using `torchcodec`.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-009 | Add markdown cell explaining Lance's blob API — first-class large binary objects that return file-like objects with lazy reads, enabling video-native operations like frame seeking | | |
| TASK-010 | Fetch a single video blob from OpenVid-1M using `ds.take_blobs("video_blob", ids=[selected_id])`. Save to a temp file and print file size. Note: verify the actual blob column name from the schema in TASK-006 — the blog uses `video_blob` but it may differ | | |
| TASK-011 | Decode the video blob using `torchcodec.decoders.VideoDecoder`. Extract a short frame range (e.g., first 10 frames), print tensor shape, and display a sample frame using matplotlib | | |
| TASK-012 | Clean up the temp video file after display | | |

### Phase 4: Vector Search via LanceDB

- GOAL-004: Run nearest-neighbor search on bundled embeddings using LanceDB from the Hub.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-013 | Add markdown cell explaining how Lance datasets can bundle embeddings and indexes as first-class citizens, enabling vector search without external infrastructure | | |
| TASK-014 | Connect to LAION-1M via `lancedb.connect("hf://datasets/lance-format/laion-1m/data")` and open the `train` table. Print schema and row count | | |
| TASK-015 | Fetch an existing row's embedding from the table (e.g., `tbl.to_lance().take([0])`) and use it as the query vector for nearest-neighbor search. Use `tbl.search(query, vector_column_name="img_emb").limit(5).to_list()`. Print captions and distances — results should be semantically similar to the seed row | | |
| TASK-016 | Add a markdown cell noting that a real CLIP model could replace the seed-row approach to enable text-to-image search as a possible extension | | |

### Phase 5: Summary & Cleanup

- GOAL-005: Add summary section and run pre-commit checks.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-017 | Add a summary markdown cell recapping the three capabilities demonstrated and linking to Lance documentation and the HF Hub Lance datasets page | | |
| TASK-018 | Add cleanup cell removing any temp files created (downloaded video blobs). No large data directories to clean up since all access was remote | | |
| TASK-019 | Run `mise run base-checks` at natural breakpoints during development. Run `mise run pre-commit` as final validation and fix all errors | | |
| TASK-020 | Test that the notebook cells execute without errors (network-dependent cells require internet access) | | |

## 3. Alternatives

- **ALT-001**: Use the FineWeb-Edu dataset (1.5B rows, text + 384-dim embeddings) instead of OpenVid-1M for the scanning demo. Rejected as the primary dataset because OpenVid-1M showcases the more distinctive blob handling feature (video). However, FineWeb-Edu is worth mentioning in the notebook as an additional exploration exercise — a markdown cell in Phase 2 could note it as an example of large-scale text+embedding scanning on the Hub.
- **ALT-002**: Download blobs locally and build a local Lance dataset from HF data. Rejected because the blog's core value proposition is remote-first access — downloading defeats the purpose of the demo.
- **ALT-003**: Use a real CLIP model to generate query embeddings for vector search. Deferred as optional extension — using an existing row's embedding (TASK-015) demonstrates the search API mechanics with meaningful results and no extra model dependencies.
- **ALT-004**: Combine this content into the existing `clip_multimodal_lance.py` notebook. Rejected because the existing notebook focuses on local Lance workflows (dataset creation, training, artifact versioning), while this notebook demonstrates remote Hub access — orthogonal concerns that benefit from separate notebooks.

## 4. Dependencies

- **DEP-001**: `pylance` — Lance format library for remote dataset access via `hf://` URIs (PyPI package `pylance` provides `import lance`)
- **DEP-002**: `lancedb` — LanceDB for vector search on Hub-hosted tables
- **DEP-003**: `pyarrow` — Arrow interop for schema inspection and table operations
- **DEP-004**: `torchcodec` — video blob decoding to torch tensors
- **DEP-005**: `torch` — tensor operations and torchcodec dependency
- **DEP-006**: `Pillow` — image display for video frames
- **DEP-007**: `numpy` — array conversion for matplotlib display
- **DEP-008**: `matplotlib` — visualization of video frames and search results

## 5. Files

- **FILE-001**: `notebooks/lance/lance_huggingface_hub.py` — new notebook (source of truth, Jupytext percent format)
- **FILE-002**: `notebooks/lance/lance_huggingface_hub.ipynb` — generated by `mise run nb:sync` (do not edit directly)

## 6. Testing

- **TEST-001**: `mise run pre-commit` passes cleanly (lint, typecheck, sync, stripout, secrets, links, markdown format)
- **TEST-002**: Notebook cells execute without import errors when dependencies are installed
- **TEST-003**: Remote dataset opens successfully and returns valid schema and row count (requires network)
- **TEST-004**: Scanner with column selection and filter returns expected metadata fields without downloading blobs
- **TEST-005**: Blob fetch returns a file-like object with non-zero bytes that can be saved and decoded
- **TEST-006**: LanceDB vector search returns results with expected fields (caption, distance)

## 7. Risks & Assumptions

- **RISK-001**: Remote HF Hub datasets may be temporarily unavailable or rate-limited. Mitigate by adding error handling around `lance.dataset()` calls with descriptive error messages suggesting retry.
- **RISK-002**: The `hf://` URI scheme and blob API require specific `pylance` / `lancedb` versions. Pin minimum versions and test against latest stable. If the integration is too new, some APIs may not yet be released — verify at implementation time.
- **RISK-003**: The actual column names and blob column names in the HF datasets may differ from the blog post examples. Verify by reading the schema at implementation time (TASK-006 schema inspection).
- **RISK-004**: Video blob download may be slow or large. Mitigate by limiting to a single blob and printing file size before decode.
- **ASSUMPTION-001**: The `lance-format/Openvid-1M` and `lance-format/laion-1m` datasets are publicly accessible on HF Hub without authentication.
- **ASSUMPTION-002**: The `hf://` URI scheme is supported by the versions of `pylance` and `lancedb` available on PyPI at implementation time.
- **ASSUMPTION-003**: `torchcodec` can decode video blobs from Lance's file-like blob objects (or from bytes written to a temp file). macOS ARM is confirmed supported (wheels published for macOS 11.0+/12.0+ ARM64).
- **ASSUMPTION-004**: `pylance>=4.0` includes `hf://` URI support natively via the Rust backend (OpenDAL). No separate `huggingface_hub` install is needed for public datasets. If authentication is required for private datasets, `huggingface_hub` may be needed — but all datasets used in this notebook are public.
- **ASSUMPTION-005**: Tests (TEST-003 through TEST-006) require network access and are executed manually in a connected environment, not in offline CI.

## 8. Related Specifications / Further Reading

- [Lance x Hugging Face Blog Post](https://lancedb.com/blog/lance-x-huggingface-a-new-era-of-sharing-multimodal-data/) — source material for this notebook
- [Lance SDK Documentation](https://lance.org/sdk_docs/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Hugging Face Lance Datasets](https://huggingface.co/lance-format) — organization hosting the demo datasets
- [torchcodec GitHub](https://github.com/meta-pytorch/torchcodec)
- [Existing CLIP notebook](../notebooks/lance/clip_multimodal_lance.py) — complementary local Lance workflow
- [Existing ColPali notebook](../notebooks/lance/colpali_vision_retriever.py) — complementary LanceDB retrieval notebook
- [Plan 003: Lance multimodal ML](003-lance-multimodal-ml.md) — related plan for the local Lance notebook
