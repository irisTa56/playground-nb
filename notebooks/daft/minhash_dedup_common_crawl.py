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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/daft/minhash_dedup_common_crawl.ipynb)
#
# # MinHash Deduplication — Common Crawl Web Text
#
# Near-duplicate detection and removal on Common Crawl web pages using
# **MinHash signatures** and **Locality-Sensitive Hashing (LSH)** with **Daft**.
#
# Based on the [Daft tutorial: MinHash Deduplication on Common Crawl](https://github.com/Eventual-Inc/Daft/blob/main/tutorials/minhash_dedupe/minhash_dedupe_common_crawl.ipynb),
# adapted for Daft 0.7+ APIs.
#
# This notebook is **self-contained**: it installs its own dependencies and
# reads data from S3 — no prior setup or credentials are needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "daft[aws]>=0.7",
#   "selectolax>=0.4",
#   "scipy>=1.17",
#   "igraph>=1.0",
#   "pandas>=3.0",
#   "matplotlib>=3.10",
#   "ipywidgets>=8.1",
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
for pkg in ("daft", "selectolax", "scipy", "igraph", "pandas", "matplotlib"):
    print(f"  {pkg}: {importlib.metadata.version(pkg)}")

# %% [markdown]
# ## 1. Configuration
#
# Key parameters for the MinHash LSH deduplication pipeline.

# %%
import os

os.environ["DAFT_PROGRESS_BAR"] = "0"

NUM_ROWS = 500  # WARC records to process (100K takes ~4min on M2)
K = 64  # number of MinHash permutations (hash functions)
NGRAM_SIZE = 5  # character n-gram size for shingling
LSH_THRESHOLD = 0.7  # Jaccard similarity threshold for LSH
SEED = 42  # random seed for reproducible MinHash
CRAWL_ID = "CC-MAIN-2024-10"

print(
    f"Pipeline config: {NUM_ROWS} rows, K={K}, "
    f"ngram={NGRAM_SIZE}, threshold={LSH_THRESHOLD}"
)

# %% [markdown]
# ## 2. Load Common Crawl Data
#
# Load WARC records from Common Crawl via anonymous S3 access. Each record
# contains a web page's URL, content type, and HTTP payload.

# %%
import daft
from daft import DataType, col
from daft.functions import monotonically_increasing_id
from daft.io import IOConfig, S3Config

IO_CONFIG = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))

df = daft.datasets.common_crawl(
    CRAWL_ID,
    content="warc",
    num_files=1,
    io_config=IO_CONFIG,
    in_aws=False,
).limit(NUM_ROWS)

# Keep only HTML response pages and materialize early (like the original tutorial)
df = df.where(col("WARC-Identified-Payload-Type") == "text/html")  # type: ignore[arg-type]
df = df.collect()

print(f"Loaded {len(df)} HTML pages from {CRAWL_ID}")

# %%
df.show(5)

# %% [markdown]
# ## 3. Extract Text Blocks
#
# Strip HTTP response headers from each WARC payload to get raw HTML, then
# parse with **selectolax** to extract meaningful text blocks (paragraphs,
# headings, list items, etc.). Each block becomes a separate row for
# fine-grained deduplication.

# %%


@daft.func()
def remove_http_headers(text: str) -> str:
    """Strip HTTP response headers from decoded WARC content."""
    if text is None:
        return ""
    sep = text.find("\r\n\r\n")
    return text[sep + 4 :] if sep >= 0 else ""


@daft.func()
def extract_text_blocks(html: str) -> list[str]:
    """Extract meaningful text blocks from HTML using selectolax."""
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    blocks: list[str] = []
    for tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"):
        for node in tree.css(tag):
            text = node.text(strip=True)
            if text and len(text) > 20:
                blocks.append(text)
    return blocks


# Decode binary payload → strip HTTP headers → extract text blocks
df = df.with_column(
    "html", remove_http_headers(col("warc_content").try_decode("utf-8"))
)
df = df.where(col("html") != "")  # type: ignore[arg-type]
df = df.with_column("blocks", extract_text_blocks(col("html")))
df = df.explode("blocks")
df = df.with_column("block_id", monotonically_increasing_id())
df = df.select("block_id", col("blocks").alias("text"))
df = df.collect()

print(f"Extracted {len(df)} text blocks")

# %%
df.show(5)

# %% [markdown]
# ## 4. Normalize Text and Compute MinHash Signatures
#
# Normalize text (lowercase, remove punctuation, collapse whitespace) then
# compute MinHash signatures with K hash functions over character n-grams.
# Two blocks with similar text produce similar MinHash vectors.

# %%
df = df.with_column(
    "normalized",
    col("text").normalize(
        remove_punct=True, lowercase=True, nfd_unicode=True, white_space=True
    ),
)
df = df.where(col("normalized") != "")  # type: ignore[arg-type]
df = df.with_column(
    "minhash",
    col("normalized").minhash(
        num_hashes=K, ngram_size=NGRAM_SIZE, seed=SEED, hash_function="xxhash"
    ),
)
df = df.where(col("minhash").not_null())
df = df.collect()

print(f"Computed MinHash signatures ({K} hashes) for {len(df)} blocks")

# %%
df.select("block_id", "text", "minhash").show(3)

# %% [markdown]
# ## 5. LSH Banding — Optimal Parameters
#
# Locality-Sensitive Hashing splits each MinHash signature into **B bands** of
# **R rows**. Two documents sharing at least one identical band are candidate
# duplicates. The `optimal_param` function chooses (B, R) to minimize the
# weighted sum of false positives and false negatives at the target threshold.

# %%
from scipy.integrate import quad


def optimal_param(threshold: float, num_hashes: int) -> tuple[int, int]:
    """Find (B, R) that minimizes FP + FN area for the given threshold."""

    def _area(b: int, r: int) -> float:
        fp, _ = quad(lambda s: 1 - (1 - s**r) ** b, 0, threshold)
        fn, _ = quad(lambda s: (1 - s**r) ** b, threshold, 1)
        return fp + fn

    best = (1, num_hashes)
    best_area = float("inf")
    for b in range(1, num_hashes + 1):
        if num_hashes % b != 0:
            continue
        r = num_hashes // b
        area = _area(b, r)
        if area < best_area:
            best_area = area
            best = (b, r)
    return best


B, R = optimal_param(LSH_THRESHOLD, K)
assert B * R == K, f"B*R must equal K: {B}*{R} != {K}"
print(f"LSH parameters: B={B} bands, R={R} rows per band (B×R={B * R})")

# %% [markdown]
# ## 6. LSH Band Hashing and Candidate Pair Detection
#
# For each band, hash the R minhash values into a single signature. Two blocks
# that share any band hash are candidate near-duplicates. We self-join on
# band hashes to find all candidate pairs.

# %%


def _make_band_hasher(band_idx: int, band_size: int):
    """Create a band hash function for the given band index."""

    def _hash_band(minhash: list) -> int:
        start = band_idx * band_size
        band = tuple(int(x) for x in minhash[start : start + band_size])
        return hash((band_idx, *band))

    return _hash_band


# Build one DataFrame per band, each with (block_id, band_hash)
band_dfs = []
for b in range(B):
    band_df = df.select("block_id", "minhash").with_column(
        "band_hash",
        col("minhash").apply(_make_band_hasher(b, R), return_dtype=DataType.int64()),
    )
    band_dfs.append(band_df.select("block_id", "band_hash"))

df_bands = band_dfs[0]
for bdf in band_dfs[1:]:
    df_bands = df_bands.union_all(bdf)

# Self-join to find candidate pairs sharing a band hash
left = df_bands.select(col("block_id").alias("id_a"), col("band_hash"))
right = df_bands.select(col("block_id").alias("id_b"), col("band_hash"))
pairs = left.join(right, on="band_hash")
pairs = pairs.where(col("id_a") < col("id_b"))
pairs = pairs.select("id_a", "id_b").distinct()
pairs = pairs.collect()

print(f"Found {len(pairs)} candidate duplicate pairs")

# %%
pairs.show(5)

# %% [markdown]
# ## 7. Connected Components via igraph
#
# Build an undirected graph from candidate pairs and find connected components.
# Each component is a cluster of near-duplicates. We keep one representative
# (minimum block_id) per component.

# %%
import igraph as ig

pairs_pd = pairs.to_pandas()

if len(pairs_pd) == 0:
    print("No duplicate pairs found — all blocks are unique")
    components = []
    component_map = {}
else:
    all_ids = set(pairs_pd["id_a"]) | set(pairs_pd["id_b"])
    id_to_idx = {bid: idx for idx, bid in enumerate(sorted(all_ids))}
    idx_to_id = {idx: bid for bid, idx in id_to_idx.items()}

    edges = [
        (id_to_idx[a], id_to_idx[b])
        for a, b in zip(pairs_pd["id_a"], pairs_pd["id_b"], strict=True)
    ]

    g = ig.Graph(n=len(id_to_idx), edges=edges, directed=False)
    components = g.connected_components(mode="weak")

    # Map each block_id to its component representative (min block_id)
    component_map = {}
    for comp in components:
        member_ids = [idx_to_id[idx] for idx in comp]
        representative = min(member_ids)
        for mid in member_ids:
            component_map[mid] = representative

    print(f"Found {len(components)} duplicate clusters from {len(all_ids)} blocks")
    sizes = [len(c) for c in components]
    print(
        f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
        f"median={sorted(sizes)[len(sizes) // 2]}"
    )

# %% [markdown]
# ## 8. Deduplicate
#
# Remove near-duplicate text blocks by keeping only the representative
# (minimum block_id) from each connected component. Blocks not part of
# any duplicate cluster are kept as-is.

# %%
original_count = len(df)

if component_map:
    # Build a lookup DataFrame: block_id → representative
    comp_df = daft.from_pydict(
        {
            "block_id": list(component_map.keys()),
            "representative": list(component_map.values()),
        }
    )
    df_with_comp = df.join(comp_df, on="block_id", how="left")
    # Keep blocks that are either not in any cluster or are the representative
    df_deduped = df_with_comp.where(
        col("representative").is_null() | (col("block_id") == col("representative"))
    )
    df_deduped = df_deduped.select("block_id", "text")
else:
    df_deduped = df.select("block_id", "text")

df_deduped = df_deduped.collect()
deduped_count = len(df_deduped)
removed = original_count - deduped_count

print(f"Original blocks:     {original_count}")
print(f"Deduplicated blocks: {deduped_count}")
print(f"Removed:             {removed} ({removed / original_count * 100:.1f}%)")

# %%
df_deduped.show(5)

# %% [markdown]
# ## 9. Duplicate Distribution
#
# Visualize the distribution of duplicate cluster sizes and the overall
# deduplication result.

# %%
import matplotlib.pyplot as plt

if components:
    cluster_sizes = [len(c) for c in components]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of cluster sizes
    ax1.hist(
        cluster_sizes,
        bins=range(1, max(cluster_sizes) + 2),
        edgecolor="black",
        alpha=0.7,
    )
    ax1.set_xlabel("Cluster size (number of near-duplicates)")
    ax1.set_ylabel("Number of clusters")
    ax1.set_title("Duplicate Cluster Size Distribution")

    # Before/after bar chart
    ax2.bar(
        ["Before", "After"],
        [original_count, deduped_count],
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black",
    )
    ax2.set_ylabel("Number of text blocks")
    ax2.set_title("Deduplication Results")
    for i, v in enumerate([original_count, deduped_count]):
        ax2.text(
            i,
            v + max(original_count, deduped_count) * 0.02,
            str(v),
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()
else:
    print("No duplicates found — skipping visualization")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# - **Common Crawl ingestion** via anonymous S3 access to WARC files
# - **HTML parsing** with selectolax to extract meaningful text blocks
# - **Text normalization** using Daft's built-in `.normalize()` method
# - **MinHash signatures** for efficient near-duplicate fingerprinting
# - **Locality-Sensitive Hashing** with optimal (B, R) parameters via scipy
# - **Self-join** on band hashes to detect candidate duplicate pairs
# - **Connected components** via igraph to cluster near-duplicates
# - **Deduplication** by keeping one representative per cluster
