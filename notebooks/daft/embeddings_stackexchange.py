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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/daft/embeddings_stackexchange.ipynb)
#
# # Semantic Embeddings — StackExchange Questions
#
# Compute sentence embeddings on the
# [RedPajama StackExchange](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
# crawl and find semantically similar questions using **Daft** and
# **SentenceTransformers**.
#
# Based on the [Daft tutorial: Embeddings on StackExchange](https://github.com/Eventual-Inc/Daft/blob/main/tutorials/embeddings/daft_tutorial_embeddings_stackexchange.ipynb),
# adapted for Daft 0.7+ APIs.
#
# This notebook is **self-contained**: it installs its own dependencies and
# reads data from S3 — no prior setup or credentials are needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "daft[aws]>=0.7",
#   "sentence-transformers>=5.3",
#   "torch>=2.11",
#   "torchvision>=0.26",
#   "accelerate>=1.13",
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
for pkg in ("daft", "sentence-transformers", "torch"):
    print(f"  {pkg}: {importlib.metadata.version(pkg)}")

# %%
import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## 1. Load the Dataset
#
# We load a StackExchange sample (~75 MB) from the RedPajama-1T crawl stored
# in S3. Each record contains the page `text` and a `meta` struct with URL and
# question score.

# %%
import os

os.environ["DAFT_PROGRESS_BAR"] = "0"

import daft
from daft.io import IOConfig, S3Config

MAX_ROWS = 10_000
MODEL_NAME = "all-MiniLM-L6-v2"

IO_CONFIG = IOConfig(s3=S3Config(anonymous=True))

df = daft.read_json(
    "s3://daft-public-data/redpajama-1t-sample/stackexchange_sample.jsonl",
    io_config=IO_CONFIG,
)
df = df.limit(MAX_ROWS)

print(f"Loading up to {MAX_ROWS} StackExchange questions")

# %%
df.show(5)

# %% [markdown]
# ## 2. Extract Metadata
#
# The `meta` column is a struct containing `url` and `question_score`. We
# extract these into top-level columns for easier access later.

# %%
df = df.with_column("url", df["meta"].get("url"))
df = df.with_column("question_score", df["meta"].get("question_score"))

df = df.select("text", "url", "question_score")
df = df.collect()

print(f"Loaded {len(df)} questions")

# %%
df.show(5)

# %% [markdown]
# ## 3. Compute Embeddings
#
# We use `SentenceTransformer` with the lightweight `all-MiniLM-L6-v2` model
# to encode each question's text into a dense vector. The model is loaded once
# and moved to the detected device.

# %%
from math import ceil, sqrt

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME, device=str(DEVICE))


@daft.func(return_dtype=daft.DataType.python())
def encode_text(text: str):
    return model.encode(text)


df = df.with_column("embedding", encode_text(df["text"]))
df = df.collect()

print(f"Computed embeddings for {len(df)} questions")

# %% [markdown]
# ## 4. Semantic Similarity Search
#
# We pick the highest-scoring questions as "top questions", then find the most
# semantically similar top question for every row in the dataset. This links
# low-visibility questions to well-known ones with similar content.

# %%
top_df = df.sort("question_score", desc=True)
top_n = ceil(sqrt(len(df)))
top_df = top_df.limit(top_n).collect()

top_questions = top_df.to_pydict()

print(f"Selected {top_n} top questions for similarity search")

# %%
from sentence_transformers.util import semantic_search


@daft.func(
    return_dtype=daft.DataType.struct(
        {
            "related_top_question": daft.DataType.string(),
            "similarity": daft.DataType.float64(),
        }
    ),
)
def find_similar(embedding):
    query = torch.tensor(embedding).unsqueeze(0)
    corpus_embeddings = torch.stack(
        [torch.tensor(e) for e in top_questions["embedding"]]
    )

    results = semantic_search(query, corpus_embeddings, top_k=1)
    best = results[0][0]
    idx = int(best["corpus_id"])
    return {
        "related_top_question": top_questions["url"][idx],
        "similarity": float(best["score"]),
    }


df = df.with_column("match", find_similar(df["embedding"]))
df = df.with_column(
    "related_top_question",
    df["match"].get("related_top_question"),
)
df = df.with_column("similarity", df["match"].get("similarity"))
df = df.select("url", "question_score", "related_top_question", "similarity")

df = df.collect()

print(f"Found semantic matches for {len(df)} questions")

# %% [markdown]
# ## 5. Inspect Results
#
# Filter out near-duplicates (similarity ≥ 0.99) and display the best matches.

# %%
results = df.where(df["similarity"] < 0.99).sort("similarity", desc=True)  # type: ignore[arg-type]
results = results.collect()

print(f"Showing {len(results)} matches (excluding near-duplicates)")

# %%
results.show(20)

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# - **Loading JSONL from S3** with anonymous access via `daft.read_json()`
# - **Struct field extraction** using `.get()` for nested metadata
# - **Sentence embeddings** with `SentenceTransformer` on GPU/CPU
# - **Semantic similarity search** linking low-visibility questions to
#   high-scoring ones
# - **UDFs in Daft** — `@daft.func` for row-wise operations and struct
#   return types
