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

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/lance/lance_huggingface_hub.ipynb)
#
# # Lance x Hugging Face Hub
#
# Remote-first access to Lance datasets hosted on
# [Hugging Face Hub](https://huggingface.co/lance-format), based on the
# [Lance x Hugging Face blog post](https://lancedb.com/blog/lance-x-huggingface-a-new-era-of-sharing-multimodal-data/).
#
# This notebook demonstrates three capabilities:
#
# 1. **Remote dataset access** — open Lance datasets via `hf://` URIs, scan
#    metadata with column selection and filters without downloading full blobs.
# 2. **Blob handling** — fetch large binary objects (video) on demand using
#    Lance's blob API, decode with `torchcodec`.
# 3. **Vector search** — run nearest-neighbor search on bundled embeddings via
#    LanceDB, directly from the Hub.
#
# This complements the existing `clip_multimodal_lance.py` (local Lance workflow)
# and `colpali_vision_retriever.py` (LanceDB retrieval) by showcasing
# **remote-first** access patterns.
#
# **Note:** This notebook requires network access — all datasets are accessed
# remotely from Hugging Face Hub. No data is downloaded in bulk.
# Anonymous access is rate-limited; set the `HF_TOKEN` environment variable
# with a [Hugging Face token](https://huggingface.co/settings/tokens) to avoid
# 429 errors.
#
# This notebook is **self-contained**: it installs its own dependencies — no
# prior setup is needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pylance>=4.0",
#   "lancedb>=0.30",
#   "pyarrow>=23.0",
#   "ipython>=9.2",
# ]
# ///

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

# %%
import importlib.metadata

print(f"Python {sys.version}")
for pkg in ("pylance", "lancedb", "pyarrow"):
    print(f"  {pkg}: {importlib.metadata.version(pkg)}")

# %% [markdown]
# ## 1. Remote Dataset Access — Metadata Scanning & Filtering
#
# Lance datasets on Hugging Face Hub can be opened directly via `hf://` URIs.
# The Lance format uses range reads under the hood, so only the requested
# columns and rows are fetched — no need to download the entire dataset.
# Blob columns (e.g., video data) are skipped unless explicitly requested,
# making metadata-only scans extremely efficient.
#
# This is a major improvement over the traditional workflow of downloading
# an entire dataset archive, extracting it locally, and then filtering — with
# Lance, you filter first and fetch only what you need.

# %%
import lance

ds = lance.dataset("hf://datasets/lance-format/Openvid-1M/data/train.lance")

print(f"Schema:\n{ds.schema}\n")
print(f"Row count: {ds.count_rows():,}")
print(f"Columns: {ds.schema.names}")

# %%
results = (
    ds.scanner(
        columns=["caption", "aesthetic_score"],
        filter="aesthetic_score >= 4.5",
        limit=10,
    )
    .to_table()
    .to_pylist()
)

print("Rows with aesthetic_score >= 4.5 (showing 10):\n")
for row in results:
    print(f"  score={row['aesthetic_score']:.2f}  {row['caption'][:80]}")

# %% [markdown]
# Notice that the query above only transferred the `caption` and
# `aesthetic_score` columns — the video blob data was never fetched. In a
# traditional workflow, you would need to download the entire ~1M-row dataset
# (including all video files) before you could filter by aesthetic score.
#
# **Explore further:** The
# [FineWeb-Edu](https://huggingface.co/datasets/lance-format/fineweb-edu-10bt)
# dataset (1.5B rows with text and 384-dim embeddings) is another excellent
# example of large-scale scanning on the Hub — try opening it with the same
# `hf://` URI pattern.

# %% [markdown]
# ## 2. Blob Handling — Video Access & Decoding
#
# Lance treats large binary objects (blobs) as first-class citizens. Blobs are
# stored separately from metadata and fetched lazily — `take_blobs` returns
# `BlobFile` objects (file-like, `io.RawIOBase`) that read data on demand.
# This means you can fetch a single video out of a million-row dataset without
# downloading anything else.

# %%
import tempfile

# The blog uses "video_blob" — verify it exists in the schema
blob_col = "video_blob"
assert blob_col in ds.schema.names, f"{blob_col!r} not in schema: {ds.schema.names}"

# Fetch row 0's caption and video blob.
# BlobFile is a file-like object (io.RawIOBase) — call .readall() for raw bytes.
row_id = 0
selected_caption = (
    ds.scanner(columns=["caption"], limit=1).to_table().to_pylist()[0]["caption"]
)
blobs = ds.take_blobs(blob_col, ids=[row_id])
with blobs[0] as blob_file:
    tmp_dir = tempfile.mkdtemp()
    tmp_video_path = Path(tmp_dir) / "sample.mp4"
    tmp_video_path.write_bytes(blob_file.readall())
print(f"Video blob saved to temp file: {tmp_video_path.stat().st_size / 1024:.1f} KB")

# %%
from IPython.display import Video, display

print(f"Caption: {selected_caption}")
display(Video(str(tmp_video_path), embed=True))

# %%
# Clean up temp video file and directory
import shutil

shutil.rmtree(tmp_dir, ignore_errors=True)

# %% [markdown]
# ## 3. Vector Search via LanceDB
#
# Lance datasets can bundle embeddings and indexes as first-class columns,
# enabling vector search without any external infrastructure. LanceDB reads
# these indexes directly from the Hub-hosted dataset, so you can run
# nearest-neighbor queries without downloading the full dataset or standing
# up a separate vector database.

# %%
import lancedb

db = lancedb.connect("hf://datasets/lance-format/laion-1m/data")
tbl = db.open_table("train")

print(f"Schema:\n{tbl.schema}\n")
print(f"Row count: {tbl.count_rows():,}")

# %%
# Use an existing row's embedding as the query vector — this avoids loading
# a separate CLIP model while still producing meaningful results (the nearest
# neighbors should be semantically similar to the seed row).
seed_row = tbl.to_lance().take([0]).to_pylist()[0]
query_vector = seed_row["img_emb"]
print(f"Seed caption: {seed_row.get('caption', 'N/A')}")
print(f"Query vector dimensions: {len(query_vector)}")

# Over-fetch and deduplicate by caption — web-scraped datasets like LAION often
# contain duplicate entries, so a small top-k may return identical rows.
raw_results = tbl.search(query_vector, vector_column_name="img_emb").limit(20).to_list()

seen_captions: set[str] = set()
unique_results: list[dict] = []
for r in raw_results:
    cap = r.get("caption", "")
    if cap not in seen_captions:
        seen_captions.add(cap)
        unique_results.append(r)
    if len(unique_results) == 5:
        break

print(
    f"\nNearest neighbors ({len(unique_results)} unique from top {len(raw_results)}):"
)
for i, r in enumerate(unique_results):
    caption = r.get("caption", "N/A")[:80]
    dist = r.get("_distance", float("nan"))
    print(f"  {i + 1}. dist={dist:.4f}  {caption}")

# %% [markdown]
# The results above should be semantically similar to the seed row — they share
# visual or conceptual features captured by the CLIP embedding space.
#
# **Extension:** To enable text-to-image search, you could replace the seed-row
# approach with a real CLIP model (e.g., `openai/clip-vit-base-patch32`) to
# generate query embeddings from arbitrary text. The search API call remains
# the same — only the query vector source changes.

# %% [markdown]
# ## Summary
#
# This notebook demonstrated three key capabilities of the Lance x Hugging Face
# Hub integration:
#
# 1. **Remote metadata scanning** — opened a million-row dataset via `hf://`
#    URI and filtered metadata without downloading blobs.
# 2. **Blob handling** — fetched a video blob on demand and played it inline.
# 3. **Vector search** — ran nearest-neighbor search on bundled CLIP embeddings
#    via LanceDB, directly from the Hub.
#
# All access was remote — no bulk downloads, no local copies, no external
# infrastructure.
#
# **Further reading:**
#
# - [Lance SDK Documentation](https://lance.org/sdk_docs/)
# - [LanceDB Documentation](https://lancedb.github.io/lancedb/)
# - [Hugging Face Lance Datasets](https://huggingface.co/lance-format)
# - [Lance x Hugging Face Blog Post](https://lancedb.com/blog/lance-x-huggingface-a-new-era-of-sharing-multimodal-data/)
