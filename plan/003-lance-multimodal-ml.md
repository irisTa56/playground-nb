---
goal: Add a Lance multimodal ML notebook covering dataset creation, CLIP training, and artifact management
version: "1.1"
date_created: 2026-03-30
last_updated: 2026-03-30
change_log:
  - date: 2026-03-30
    version: "1.1"
    summary: >
      Implementation complete. Fixed PyPI package name (lance → pylance),
      num_workers=0 for Lance fork safety, data paths under data/,
      removed dead residual connection in ProjectionHead, added Arrow
      schema interoperability note.
  - date: 2026-03-30
    version: "1.0"
    summary: >
      Initial plan.
owner: takayuki
status: Done
tags:
  - feature
  - notebook
  - lance
  - multimodal
  - clip
---

# Introduction

![Status: Done](https://img.shields.io/badge/status-Done-brightgreen)

Add a single self-contained notebook `notebooks/lance/clip_multimodal_lance.py` that walks through a complete multimodal ML workflow using **Lance** as the columnar data format:

1. **Dataset creation** — download Flickr8k, parse annotations, store images + captions in a Lance dataset
2. **CLIP training** — build image/text encoders, train with contrastive loss, reading data from Lance
3. **Artifact management** — save/load/version PyTorch model checkpoints as Lance datasets

This consolidates the three official Lance examples into one cohesive notebook following this repo's conventions (PEP 723, self-contained setup, Jupytext percent format).

## 1. Requirements & Constraints

- **REQ-001**: Cover all three Lance examples: Flickr8k dataset creation, CLIP training, artifact versioning.
- **REQ-002**: Use Lance (not LanceDB) as the core storage layer — `lance.write_dataset()` / `lance.dataset()`.
- **SEC-001**: No hardcoded credentials or API keys. Flickr8k data must be downloaded from a public source.
- **CON-001**: GPU/MPS recommended for training. CPU is supported but impractically slow. Verification will be done on MPS or GPU.
- **CON-002**: Memory-efficient dataset creation using PyArrow generator/RecordBatchReader pattern.
- **PAT-001**: Use `pa.schema()` with explicit types for all Lance writes.
- **PAT-002**: Use `lance.dataset(path, version=N)` to demonstrate version-based artifact loading.
- **PAT-003**: Link to original Lance examples in notebook header markdown.

## 2. Implementation Steps

### Phase 1: Notebook Scaffolding

- GOAL-001: Create the notebook file with correct boilerplate and PEP 723 metadata.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Scaffold `notebooks/lance/clip_multimodal_lance.py` using `/create-notebook` skill. Add Colab badge, title/description markdown (one sentence per line), and links to the three source Lance examples | ✅ | 2026-03-30 |
| TASK-002 | Add PEP 723 `# /// script` block with dependencies: `pylance>=3.0`, `pyarrow>=23.0`, `torch>=2.11`, `torchvision>=0.26`, `timm>=1.0`, `transformers>=5.4`, `tqdm>=4.67`, `matplotlib>=3.10`, `numpy>=2.4`, `Pillow>=12.1` (verified latest stable versions on PyPI at implementation time). Note: PyPI package is `pylance` (not `lance`), which provides `import lance`. No `opencv-python-headless` — use PIL for image I/O, consistent with existing notebooks | ✅ | 2026-03-30 |
| TASK-003 | Verify `_get_deps`, `_run`, `_setup` boilerplate from `/create-notebook` matches existing notebooks. Adjust `_setup()` only if notebook-specific post-install steps are needed | ✅ | 2026-03-30 |
| TASK-004 | Add device detection cell (`torch.cuda.is_available()` / `torch.backends.mps.is_available()` / CPU fallback) with `get_device()` function | ✅ | 2026-03-30 |

### Phase 2: Flickr8k Dataset Creation (Lance)

- GOAL-002: Download Flickr8k and convert to a Lance dataset with image bytes + captions.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Add markdown cell explaining Flickr8k dataset structure (8k images, 5 captions each) and the Lance columnar format benefits | ✅ | 2026-03-30 |
| TASK-006 | Implement data download cell — download Flickr8k from `jbrownlee/Datasets` GitHub releases (`Flickr8k_Dataset.zip` + `Flickr8k_text.zip`, no authentication required). Define `DATA_DIR` and `LANCE_DIR` path constants | ✅ | 2026-03-30 |
| TASK-007 | Implement annotation parsing — read caption file, group 5 captions per image into a dict `{image_id: [cap1, ..., cap5]}` | ✅ | 2026-03-30 |
| TASK-008 | Define `pa.schema()` with columns: `image_id` (string), `image` (binary), `captions` (list of string) | ✅ | 2026-03-30 |
| TASK-009 | Implement generator function that yields `pa.RecordBatch` objects (batch size ~100–500 rows) — read each image as bytes via `Path.read_bytes()`, pair with captions | ✅ | 2026-03-30 |
| TASK-010 | Write Lance dataset using `pa.RecordBatchReader.from_batches(schema, generator)` piped to `lance.write_dataset()` | ✅ | 2026-03-30 |
| TASK-011 | Add verification cell — open the Lance dataset, print row count, sample a row, display a sample image with its captions using `matplotlib` | ✅ | 2026-03-30 |

### Phase 3: CLIP Model Definition & Training

- GOAL-003: Define CLIP architecture and train on the Flickr8k Lance dataset.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-012 | Add markdown cell explaining CLIP contrastive learning (image encoder + text encoder → shared embedding space, trained with InfoNCE loss) | ✅ | 2026-03-30 |
| TASK-013 | Define `Config` dataclass with hyperparameters: `image_size=224`, `batch_size=32`, `embed_dim=256`, `epochs=3`, `lr_head=1e-3`, `lr_image=1e-4`, `lr_text=1e-5`. Temperature is a learnable parameter initialized to `log(1/0.07)` per the original CLIP paper (not a fixed hyperparameter) | ✅ | 2026-03-30 |
| TASK-014 | Implement `CLIPLanceDataset(torch.utils.data.Dataset)` — reads from `lance.dataset(path)`, uses `ds.take([idx])` for image bytes, `ds.count_rows()` for length, decodes image via `PIL.Image.open(io.BytesIO(...))`, applies `torchvision.transforms` (Resize, CenterCrop, Normalize), tokenizes longest caption with `AutoTokenizer("bert-base-cased")`. Note: `ds.take` random access is adequate here; scanner-based batching is an option if perf is a bottleneck | ✅ | 2026-03-30 |
| TASK-015 | Implement `ImageEncoder(nn.Module)` — uses `timm.create_model("resnet50", pretrained=True, num_classes=0)` for feature extraction (outputs 2048-dim pooled features), no manual FC replacement needed | ✅ | 2026-03-30 |
| TASK-016 | Implement `TextEncoder(nn.Module)` — uses `AutoModel.from_pretrained("bert-base-cased")`, extracts CLS token embedding | ✅ | 2026-03-30 |
| TASK-017 | Implement `ProjectionHead(nn.Module)` — Linear → GELU → Linear → LayerNorm, projects to `embed_dim` (no residual — `in_dim ≠ out_dim`) | ✅ | 2026-03-30 |
| TASK-018 | Implement `CLIPModel(nn.Module)` — combines ImageEncoder, TextEncoder, two ProjectionHeads, and a learnable `log_temperature` parameter (initialized to `log(1/0.07)`, clamped to `log(100)` max). Forward returns normalized image/text embeddings | ✅ | 2026-03-30 |
| TASK-019 | Implement CLIP contrastive loss function — compute cosine similarity matrix, cross-entropy loss on both image→text and text→image directions, average | ✅ | 2026-03-30 |
| TASK-020 | Implement training loop — DataLoader, AdamW optimizer with separate param groups (head/image/text learning rates), epoch loop with tqdm progress bar, print loss per epoch | ✅ | 2026-03-30 |
| TASK-021 | Run training for configured epochs, print final loss | ✅ | 2026-03-30 |

### Phase 4: Artifact Management (Model Versioning with Lance)

- GOAL-004: Save trained model checkpoints as versioned Lance datasets and demonstrate loading.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-022 | Add markdown cell explaining Lance's built-in versioning for model artifacts (each `mode="overwrite"` creates a new version, old versions remain accessible) | ✅ | 2026-03-30 |
| TASK-023 | Define `pa.schema()` for model storage: `name` (string), `value` (list of float64), `shape` (list of int64) | ✅ | 2026-03-30 |
| TASK-024 | Implement `save_model(state_dict, path)` — generator that flattens each parameter tensor to 1D, records shape, yields RecordBatch per parameter. Write with `lance.write_dataset(reader, path, mode="overwrite")` | ✅ | 2026-03-30 |
| TASK-025 | Save the trained CLIP model after each epoch during training (TASK-020). This naturally produces multiple Lance versions (one per epoch) for the versioning demo | ✅ | 2026-03-30 |
| TASK-026 | Implement `load_model(path, version)` — open `lance.dataset(path, version=N)`, iterate rows, reshape flat arrays back to tensors using stored shapes, return `OrderedDict` | ✅ | 2026-03-30 |
| TASK-027 | Demonstrate loading version 1 (epoch 1) vs latest version (final epoch) — load both, compare a parameter tensor to show they differ, load version 1 into a fresh model and verify `model.load_state_dict()` succeeds | ✅ | 2026-03-30 |

### Phase 5: Cleanup & Validation

- GOAL-005: Add cleanup cell, run checks at natural breakpoints, ensure notebook executes end-to-end.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-028 | Add cleanup cell — remove downloaded data and Lance datasets (`shutil.rmtree`) with `if Path(...).exists()` guards | ✅ | 2026-03-30 |
| TASK-029 | Run `mise run base-checks` at natural breakpoints during each phase (lint, typecheck, format). Run `mise run pre-commit` as the final validation and fix all errors | ✅ | 2026-03-30 |
| TASK-030 | Verify notebook runs end-to-end (at least the non-training cells) without errors | ✅ | 2026-03-30 |

## 3. Alternatives

- **ALT-001**: Split into three separate notebooks (one per Lance example). Rejected because the three examples form a natural pipeline (create dataset → train model → manage artifacts) and a single notebook better demonstrates the end-to-end workflow.
- **ALT-002**: Use LanceDB instead of Lance. Rejected because the source examples use the lower-level `lance` library directly, and the existing `colpali_vision_retriever.py` already covers LanceDB — this notebook should showcase the complementary `lance` API.
- **ALT-003**: Use HuggingFace `datasets` to load Flickr8k directly. Rejected — using `jbrownlee/Datasets` GitHub releases instead (direct zip download, no extra dependencies). The dataset creation phase demonstrates Lance's write path by converting the raw files into a Lance dataset.

## 4. Dependencies

- **DEP-001**: `pylance` — core columnar format library for dataset creation, reading, and versioning (PyPI package `pylance` provides `import lance`)
- **DEP-002**: `pyarrow` — schema definition, RecordBatch construction, RecordBatchReader
- **DEP-003**: `torch` + `torchvision` — model definition, training loop, transforms
- **DEP-004**: `timm` — pre-trained ResNet50 image encoder
- **DEP-005**: `transformers` — BERT tokenizer and text encoder
- **DEP-006**: `Pillow` — image reading/decoding from bytes (consistent with existing notebooks; replaces opencv)
- **DEP-007**: `tqdm` — progress bars for dataset creation and training
- **DEP-008**: `matplotlib` — sample image visualization
- **DEP-009**: `numpy` — array manipulation for image decoding

## 5. Files

- **FILE-001**: `notebooks/lance/clip_multimodal_lance.py` — new notebook (source of truth, Jupytext percent format)
- **FILE-002**: `notebooks/lance/clip_multimodal_lance.ipynb` — generated by `mise run nb:sync` (do not edit directly)

## 6. Testing

- **TEST-001**: `mise run pre-commit` passes cleanly (lint, typecheck, sync, stripout, secrets, links, markdown format)
- **TEST-002**: Notebook cells execute without import errors when dependencies are installed
- **TEST-003**: Lance dataset is created with correct schema and expected row count (~8,000 rows for Flickr8k)
- **TEST-004**: Training loop runs for at least a few mini-batches without errors on MPS/GPU (CPU is supported but impractically slow per CON-001; CPU smoke-test should limit to ~5 batches)
- **TEST-005**: Model save/load round-trip preserves parameter values (version 1 loaded == original state_dict)
- **TEST-006**: Lance versioning works — version 1 and version 2 are distinct and independently loadable

## 7. Risks & Assumptions

- **RISK-001**: Flickr8k download source availability — mitigated by using `jbrownlee/Datasets` GitHub releases (direct zip download, no authentication required).
- **RISK-002**: Dataset size (~1 GB images) may be slow to download in Colab. Mitigate by adding a progress bar and documenting expected download time.
- **RISK-003**: CLIP training requires GPU/MPS for practical execution. CPU fallback is supported but not verified.
- **RISK-004**: `pylance` API may have breaking changes between versions. Mitigate by pinning minimum version and testing against latest stable. Note: PyPI package is `pylance` (not `lance` — that's a different package).
- **ASSUMPTION-001**: Flickr8k has ~8,091 images with 5 captions each — annotation parsing logic assumes this structure.
- **ASSUMPTION-002**: `timm` ResNet50 and `transformers` BERT-base-cased are available without authentication.
- **ASSUMPTION-003**: The notebook will be executed with sufficient disk space (~2 GB for data + Lance datasets + model artifacts).

## 8. Related Specifications / Further Reading

- [Lance Flickr8k Dataset Creation](https://lance.org/examples/python/flickr8k_dataset_creation/)
- [Lance CLIP Training](https://lance.org/examples/python/clip_training/)
- [Lance Artifact Management](https://lance.org/examples/python/artifact_management/)
- [Lance SDK Docs](https://lance.org/sdk_docs/)
- [CLIP Paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [Existing ColPali notebook](../notebooks/lance/colpali_vision_retriever.py) — complementary LanceDB-based notebook in this repo
