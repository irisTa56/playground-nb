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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/lance/colpali_vision_retriever.ipynb)
#
# # ColPali Vision Retriever
#
# Late interaction & efficient multi-modal retrievers using
# [ColPali](https://huggingface.co/vidore/colpali-v1.2) and
# [LanceDB](https://lancedb.com/).
#
# **Codebase** — <https://github.com/AyushExel/vision-retrieval>
# (based on <https://github.com/kyryl-opens-ml/vision-retrieval>)
#
# ColPali is a visual retriever model that combines:
#
# * **PaliGemma** — a VLM with SigLIP-So400m/14 vision encoder + Gemma-2B
#   language model, plus projection layers that map to 128-dim vectors.
# * **Late interaction** mechanism based on ColBERT.
#
# It works in two phases:
#
# * **Offline** — each document page is processed as image patches through the
#   vision encoder. Each of the 1030 patches is projected to a 128-dim vector
#   and stored as a multi-vector representation.
# * **Online** — the user query is encoded via the language model, then MaxSim
#   scores are computed against stored patch vectors. Top-K pages are returned.
#
# This notebook is **self-contained**: it installs its own dependencies, downloads
# the dataset, and detects the available accelerator — no prior setup is needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.10",
#   "torchvision>=0.25",
#   "colpali-engine==0.2.2",
#   # Pin to a version colpali-engine 0.2.2 was tested against; newer
#   # versions change PaliGemmaProcessor <image> token handling and
#   # degrade retrieval accuracy.
#   "transformers==4.44.2",
#   "lancedb>=0.30",
#   "tantivy>=0.25",
#   "pillow>=12.0",
#   "pdf2image>=1.17",
#   "pypdf>=6.0",
#   "mteb>=1.12,<1.13",
#   "huggingface-hub>=0.26,<1",
#   "ipywidgets>=8.1",
# ]
# ///

import re
import shutil
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

    # pdf2image requires poppler (pdfinfo / pdftoppm)
    if shutil.which("pdfinfo") is None:
        if shutil.which("apt-get"):
            _run(["sudo", "apt-get", "install", "-y", "-qq", "poppler-utils"])
        elif shutil.which("brew"):
            _run(["brew", "install", "poppler"])
        else:
            print("Warning: poppler not found — install it manually for pdf2image")


_setup()

# %%
import importlib.metadata

print(f"Python {sys.version}")
for pkg in ("torch", "colpali-engine", "lancedb", "pillow", "huggingface-hub"):
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


print(f"Using device: {get_device()}")

# %% [markdown]
# ## Clone the vision-retrieval helper library

# %%
REPO_DIR = Path("vision-retrieval")
if not REPO_DIR.exists():
    _run(["git", "clone", "https://github.com/AyushExel/vision-retrieval.git", "-q"])
    _run(["cp", "-a", f"{REPO_DIR}/.", "."])

# %% [markdown]
# ## Authenticate with Hugging Face
#
# PaliGemma (`google/paligemma-3b-mix-448`) is a **gated model** — you need to
# [accept the licence](https://huggingface.co/google/paligemma-3b-mix-448) and
# provide an access token.
# %%
from huggingface_hub import hf_hub_download, login
from vision_retrieval.core import (  # type: ignore[import-not-found]
    create_db,
    embedd_docs,
    get_model_colpali,
    search,
)

login()  # prompts for token in Colab / reads from HF_TOKEN env var locally

# %% [markdown]
# ## Load Models
#
# Only load the ColPali model if memory is a constraint.

# %%
model_colpali, processor_colpali = get_model_colpali()  # mix-448 model
# model_phi_vision, processor_phi_vision = get_model_phi_vision()  # uncomment for Phi vision responses

# %% [markdown]
# ## Download Dataset
#
# We ingest documents from very different genres to make retrieval challenging:
#
# * Financial reports (Q2 2024): Apple, Amazon, Meta, Alphabet, Netflix, Starbucks
# * Naruto Volume 72
# * Arabian Nights
# * Children's short-story collections
# * InfraRed Cloud report

# %%
import zipfile

DATA_DIR = Path("fin_pdf_data")

if not DATA_DIR.exists():
    hf_hub_download(
        repo_id="ayushexel/pdf_colpali",
        filename="archive1.zip",
        repo_type="dataset",
        local_dir="./",
    )
    with zipfile.ZipFile("archive1.zip") as zf:
        zf.extractall(".")
    Path("archive1.zip").unlink(missing_ok=True)

print(f"PDF directory: {DATA_DIR}  ({sum(1 for _ in DATA_DIR.glob('*.pdf'))} files)")

# %% [markdown]
# ### (Optional) Remove some docs to speed up

# %%
REMOVE_PDFS = [
    "2024q2-alphabet-earnings-release.pdf",
    "AMZN-Q2-2024-Earnings-Release.pdf",
    "FINAL-Q2-24-Shareholder-Letter.pdf",
    "FY24_Q2_Consolidated_Financial_Statements.pdf",
    "Meta - Meta Reports Second Quarter 2024 Results.pdf",
    "Microsoft (MSFT) Q2 earnings report 2024.pdf",
    "Starbucks Coffee Company - Starbucks Reports Q2 Fiscal 2024 Results.pdf",
]
for name in REMOVE_PDFS:
    (DATA_DIR / name).unlink(missing_ok=True)

# Remove stray Jupyter checkpoints
for ckpt in DATA_DIR.glob(".ipynb_checkpoints"):
    shutil.rmtree(ckpt, ignore_errors=True)

# %%
docs_to_embed = embedd_docs(
    str(DATA_DIR), model=model_colpali, processor=processor_colpali
)

# %% [markdown]
# ## Ingest Data into LanceDB
#
# We stream batch iterators to LanceDB which persists data on disk — unlikely to
# OOM even though each document has high-dimensional vectors (1030 x 128).
#
# For one of the comparison experiments we also store extracted PDF text in a
# column to create an FTS/BM25 index.

# %%
DB_PATH = "lancedb"
TABLE_NAME = "collection"

tbl = create_db(docs_storage=docs_to_embed, table_name=TABLE_NAME, db_path=DB_PATH)
print(f"Ingested {tbl.count_rows()} pages into LanceDB")

# %% [markdown]
# ---
# ## Retrieval
#
# We compare three retrieval strategies:
#
# 1. **ColPali MaxSim** — full MaxSim across all ingested pages
# 2. **ColPali as FTS reranker** — pre-filter with LanceDB FTS, then rerank
# 3. **ColPali as vector-search reranker** — pre-filter with similarity search,
#    then rerank

# %% [markdown]
# ### 1. ColPali Retrieval (Full MaxSim)

# %%
import time

import lancedb
from PIL import Image


def timed_search(query: str, label: str, **kwargs) -> Image.Image:
    """Run a search query, print elapsed time, and return the top result image."""
    t1 = time.perf_counter()
    results = search(
        query=query,
        table_name=TABLE_NAME,
        model=model_colpali,
        processor=processor_colpali,
        db_path=DB_PATH,
        top_k=3,
        **kwargs,
    )
    elapsed = time.perf_counter() - t1
    print(f"{label} — {elapsed:.2f}s")
    return results[0]["pil_image"]


# %%
timed_search("How do model training costs change over time?", "ColPali MaxSim")

# %% [markdown]
# ### 2. ColPali as FTS Reranker
#
# When text can be extracted from the PDF, an FTS index reduces the search space
# (here to the top 100 matches) before ColPali reranks.

# %%
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)
table.create_fts_index("page_text", replace=True)

# %%
timed_search(
    "Why obito wanted kakashi to become the sixth hokage?",
    "ColPali + FTS rerank",
    fts=True,
)

# %% [markdown]
# ### 3. ColPali as Vector-Search Reranker
#
# A vision retriever ideally shouldn't depend on text extraction. Instead we
# flatten the 128-dim patch embeddings, zero-pad the query to match document
# dimensions, and run vector search to pre-filter before MaxSim reranking.

# %%
timed_search(
    "Why obito wanted kakashi to become the sixth hokage?",
    "ColPali + vector rerank",
    vector=True,
)

# %%
timed_search(
    "What did the queen ask her magic mirror everyday?",
    "ColPali + vector rerank",
    vector=True,
)

# %%
timed_search(
    "How did the fish become men, women, and children?",
    "ColPali + vector rerank",
    vector=True,
)
