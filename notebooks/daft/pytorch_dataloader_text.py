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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/daft/pytorch_dataloader_text.ipynb)
#
# # PyTorch DataLoader — Text Data
#
# IMDB sentiment classification with **Pandas**, **Polars**, and **Daft**.
#
# This notebook is **self-contained**: it installs its own dependencies, downloads
# the dataset, and detects the available accelerator — no prior setup is needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.10",
#   "pandas>=3.0",
#   "polars>=1.39",
#   "daft>=0.7",
#   "transformers>=4.40",
#   "scikit-learn>=1.8",
#   "kaggle>=2.0",
#   "numpy>=2.4",
#   "ipywidgets>=8.1",
# ]
# ///

import os
import re
import subprocess
import sys


def _get_deps():
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        src = ip.user_ns.get("In", [""])[ip.execution_count]
    except (NameError, IndexError):
        src = open(__file__).read()

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
for pkg in ("torch", "pandas", "polars", "daft", "transformers", "numpy"):
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

# %%
import zipfile
from pathlib import Path

DATA_DIR = Path("data/imdb")
CSV_PATH = DATA_DIR / "IMDB Dataset.csv"

if not CSV_PATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
                "-p",
                str(DATA_DIR),
                "--unzip",
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Kaggle CLI failed ({exc}). Download manually from:")
        print(
            "  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        )
        print(f"and extract 'IMDB Dataset.csv' to {DATA_DIR}/")

    # Kaggle sometimes downloads as zip without --unzip working
    for zf in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(DATA_DIR, filter="data")  # type: ignore[call-arg]  # Python 3.12+
        zf.unlink()

print(f"Dataset ready: {CSV_PATH}  ({CSV_PATH.stat().st_size:,} bytes)")

# %% [markdown]
# ## 1. `make_dataloader` Helper
#
# Same platform-aware helper as the tabular notebook — `num_workers=0` on macOS
# to avoid multiprocessing issues, `pin_memory=True` when using CUDA.
#
# The default `collate_fn` is set to `collate_imdb` (defined in §2) so that
# variable-length tokenised sequences are dynamically padded per batch.

# %%
from torch.utils.data import DataLoader, Dataset


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0 if sys.platform == "darwin" else (os.cpu_count() or 2) // 2,
    pin_memory: bool = DEVICE.type == "cuda",
    persistent_workers: bool | None = None,
    collate_fn=None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for text tasks."""
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


# %% [markdown]
# ## 2. `collate_fn` — Handling Variable-Length Sequences
#
# Text samples vary in length. The default `collate_fn` stacks tensors of
# **identical shape**, so for variable-length inputs we have two options:
#
# 1. **Pad in the Dataset** — tokenize with `padding="max_length"` so every
#    sample has the same shape. Simple but wastes computation on padding tokens.
# 2. **Pad in `collate_fn`** — tokenize without padding, then pad each batch
#    to the length of its longest sample. More efficient but requires a custom
#    collation function.
#
# We demonstrate both approaches below. The Dataset uses `max_length` padding
# for simplicity; the custom `collate_fn` shows the dynamic-padding alternative.

# %%


def collate_imdb(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length tokenised samples by padding to the batch max length.

    Each sample is a tuple of (input_ids, attention_mask, label).
    This function pads input_ids and attention_mask to the longest sequence in the
    batch, which is more efficient than padding every sample to ``max_length``.
    """
    input_ids_list, attention_mask_list, labels = zip(*batch, strict=True)

    # Find the longest sequence in this batch
    max_len = max(ids.size(0) for ids in input_ids_list)

    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(
        zip(input_ids_list, attention_mask_list, strict=True)
    ):
        length = ids.size(0)
        padded_ids[i, :length] = ids
        padded_mask[i, :length] = mask

    return padded_ids, padded_mask, torch.stack(labels)


# %% [markdown]
# ---
# ## 3. CSV Loading Comparison
#
# For text data the tokenizer dominates preprocessing time, but CSV parsing
# still matters for large files. We time all three libraries loading the
# same IMDB dataset.

# %%
import timeit
from collections.abc import Callable


def bench[T](fn: Callable[[], T], n: int = 3) -> tuple[T, float]:
    """Run *fn* **n** times and return ``(last_result, avg_seconds)``."""
    result: T | None = None
    total = 0.0
    for _ in range(n):
        t0 = timeit.default_timer()
        result = fn()
        total += timeit.default_timer() - t0
    assert result is not None
    return result, total / n


# %%
import pandas as pd

df_pd, t_pd = bench(lambda: pd.read_csv(CSV_PATH)[["review", "sentiment"]].dropna())
print(f"Pandas  — {len(df_pd):,} rows, {t_pd:.3f}s avg")
df_pd.head()

# %%
import polars as pl

df_pl, t_pl = bench(
    lambda: pl.read_csv(CSV_PATH).select(["review", "sentiment"]).drop_nulls()
)
print(f"Polars  — {len(df_pl):,} rows, {t_pl:.3f}s avg")
df_pl.head()

# %%
import daft

# Suppress Daft's tqdm progress bar (e.g. "Read CSV: 50,000 rows out")
os.environ["DAFT_PROGRESS_BAR"] = "0"

df_daft, t_daft = bench(
    lambda: daft.read_csv(str(CSV_PATH)).select("review", "sentiment").collect()
)
print(f"Daft    — {len(df_daft):,} rows, {t_daft:.3f}s avg")
df_daft.to_pandas().head()

# %%
print("\n--- CSV Loading Time Summary ---")
print(f"  Pandas:  {t_pd:.3f}s")
print(f"  Polars:  {t_pl:.3f}s")
print(f"  Daft:    {t_daft:.3f}s")

# %% [markdown]
# ---
# ## 4. Tokenisation with BERT
#
# We use `bert-base-uncased` (~440 MB on first download) to tokenise the
# reviews. The tokenizer converts raw text into `input_ids` (integer token
# indices) and `attention_mask` (1 for real tokens, 0 for padding).

# %%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(f"Vocab size: {tokenizer.vocab_size:,}")

# Quick demo
sample = "This movie was absolutely fantastic!"
encoded = tokenizer(sample, truncation=True, padding="max_length", max_length=16)
print(f"Input IDs:      {encoded['input_ids']}")
print(f"Attention mask:  {encoded['attention_mask']}")
print(
    f"Decoded:         {tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)}"
)

# %% [markdown]
# ## 5. `IMDBDataset`
#
# The Dataset tokenises each review on the fly in `__getitem__`.
# We use `padding=False` so each sample is its **actual length** — the
# `collate_imdb` function (defined in §2) then pads each batch to its
# longest sequence, which is more efficient than padding every sample to
# `max_length`.

# %%
from torch.utils.data import Dataset as TorchDataset


class IMDBDataset(TorchDataset):
    """Map-style dataset for IMDB sentiment classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: BertTokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoding["input_ids"].squeeze(0)
        attention_mask: torch.Tensor = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return input_ids, attention_mask, label


# %% [markdown]
# ## 6. Text & Label Extraction (All Backends)
#
# Each library loaded the same CSV. We now extract `(texts, labels)` as plain
# Python lists from all three so we can feed them into the **same** Dataset →
# DataLoader → training pipeline and compare the results.
# Labels are encoded as `positive → 1`, `negative → 0`.

# %%
from sklearn.model_selection import train_test_split


def extract_pandas(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    texts = df["review"].tolist()
    labels = (df["sentiment"] == "positive").astype(int).tolist()
    return texts, labels


def extract_polars(df: pl.DataFrame) -> tuple[list[str], list[int]]:
    texts = df["review"].to_list()
    labels = (df["sentiment"] == "positive").cast(int).to_list()
    return texts, labels


def extract_daft(df: daft.DataFrame) -> tuple[list[str], list[int]]:
    d = df.to_pydict()
    texts = d["review"]
    labels = [1 if s == "positive" else 0 for s in d["sentiment"]]
    return texts, labels


backends: dict[str, tuple[list[str], list[int]]] = {
    "Pandas": extract_pandas(df_pd),
    "Polars": extract_polars(df_pl),
    "Daft": extract_daft(df_daft),
}

for name, (txts, lbls) in backends.items():
    print(f"{name:>6} — {len(txts):,} texts, {sum(lbls):,} positive")

# %% [markdown]
# ## 7. Batch Verification
#
# Inspect a single batch to confirm tensor shapes match expectations.

# %%
# Build a quick loader from the Pandas backend for verification
_texts, _labels = backends["Pandas"]
_texts_train, _texts_test, _labels_train, _labels_test = train_test_split(
    _texts, _labels, test_size=0.2, random_state=42, stratify=_labels
)
_verify_ds = IMDBDataset(_texts_train[:64], _labels_train[:64], tokenizer)
_verify_loader = make_dataloader(
    _verify_ds, batch_size=16, shuffle=False, collate_fn=collate_imdb
)

batch_ids, batch_mask, batch_labels = next(iter(_verify_loader))
print(f"input_ids shape:      {batch_ids.shape}")  # [16, ≤128] (dynamically padded)
print(f"attention_mask shape: {batch_mask.shape}")  # [16, ≤128]
print(f"labels shape:         {batch_labels.shape}")  # [16]
print(
    f"Label distribution in batch: {batch_labels.sum().item():.0f} positive"
    f" / {len(batch_labels)} total"
)

# %% [markdown]
# ---
# ## 8. Training & Evaluation (All Backends)
#
# A lightweight classifier: BERT embeddings (frozen) → mean pooling → linear
# head. We train for a few epochs on a small subset to verify the end-to-end
# pipeline, then compare results across all three backends.

# %%
import torch.nn as nn
from transformers import BertModel


class SentimentClassifier(nn.Module):
    """Simple BERT-based binary classifier (frozen BERT + linear head)."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # Freeze BERT parameters for fast training
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over token embeddings (respecting the attention mask)
        token_embs = outputs.last_hidden_state  # [B, seq_len, hidden]
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        sum_embs = (token_embs * mask_expanded).sum(dim=1)  # [B, hidden]
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        pooled = sum_embs / count  # [B, hidden]
        return self.classifier(pooled).squeeze(-1)  # [B]


def train_and_evaluate(
    model: SentimentClassifier,
    texts_train: list[str],
    texts_test: list[str],
    labels_train: list[int],
    labels_test: list[int],
    *,
    num_epochs: int = 2,
    batch_size: int = 16,
    max_samples_train: int = 2000,
    max_samples_test: int = 500,
    lr: float = 2e-4,
    seed: int = 0,
) -> dict[str, float]:
    """Run full Dataset → DataLoader → train → eval pipeline and return metrics."""
    torch.manual_seed(seed)

    train_ds = IMDBDataset(
        texts_train[:max_samples_train],
        labels_train[:max_samples_train],
        tokenizer,
    )
    test_ds = IMDBDataset(
        texts_test[:max_samples_test],
        labels_test[:max_samples_test],
        tokenizer,
    )
    train_loader = make_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_imdb
    )
    test_loader = make_dataloader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_imdb
    )

    # Reset the classifier head so each run starts from scratch
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # --- Training ---
    train_loss = 0.0
    train_acc = 0.0
    for _epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for ids, mask, lbls in train_loader:
            ids, mask, lbls = ids.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE)
            logits = model(ids, mask)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ids.size(0)
            correct += ((logits > 0).float() == lbls).sum().item()
            total += ids.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

    # --- Evaluation ---
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for ids, mask, lbls in test_loader:
            ids, mask, lbls = ids.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE)
            logits = model(ids, mask)
            test_loss += criterion(logits, lbls).item() * ids.size(0)
            correct += ((logits > 0).float() == lbls).sum().item()
            total += ids.size(0)

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss / total,
        "test_acc": correct / total,
    }


# %%
# Instantiate once — BERT weights are loaded only here.
model = SentimentClassifier().to(DEVICE)

# Warmup: absorb one-time costs (MPS shader compilation, etc.)
_warmup_texts, _warmup_labels = next(iter(backends.values()))
_wt_tr, _wt_te, _wl_tr, _wl_te = train_test_split(
    _warmup_texts,
    _warmup_labels,
    test_size=0.2,
    random_state=42,
    stratify=_warmup_labels,
)
_ = train_and_evaluate(model, _wt_tr, _wt_te, _wl_tr, _wl_te)
print("Warmup done.\n")

results: dict[str, dict[str, float]] = {}
for name, (txts, lbls) in backends.items():
    t_tr, t_te, l_tr, l_te = train_test_split(
        txts, lbls, test_size=0.2, random_state=42, stratify=lbls
    )
    t0 = timeit.default_timer()
    metrics = train_and_evaluate(model, t_tr, t_te, l_tr, l_te)
    elapsed = timeit.default_timer() - t0
    results[name] = {**metrics, "time_sec": elapsed}
    print(
        f"{name:>6}  train_loss={metrics['train_loss']:.4f}  "
        f"train_acc={metrics['train_acc']:.3f}  "
        f"test_loss={metrics['test_loss']:.4f}  "
        f"test_acc={metrics['test_acc']:.3f}  "
        f"time={elapsed:.1f}s"
    )

# %% [markdown]
# All three backends produce **identical** results — the texts and labels
# extracted from Pandas, Polars, and Daft are the same Python lists, so the
# DataLoader sees exactly the same data. This confirms the key point from the
# tabular notebook: **the DataLoader is agnostic to the loading backend**.

# %%
results_df = pd.DataFrame(results).T
results_df.index.name = "backend"
results_df

# %% [markdown]
# ---
# ## 9. Comparison
#
# ### CSV Loading Time
#
# | Library | Approach | Strengths |
# |---------|----------|-----------|
# | Pandas | `read_csv` → `dropna` | Ubiquitous, rich ecosystem |
# | Polars | `read_csv` → `drop_nulls` | Rust-based parser, often 2–5× faster |
# | Daft | `read_csv` → `collect` | Lazy evaluation, native cloud I/O |
#
# ### Key Observations
#
# - **The tokenizer dominates preprocessing time.** Unlike tabular data where
#   encoding and scaling take measurable time, here the BERT tokenizer in
#   `__getitem__` is the bottleneck. The CSV loading library matters less.
# - **Library choice matters for large-scale loading.** For CSVs of several
#   hundred MB or more, Polars' Rust parser substantially outperforms Pandas.
#   Daft shines when reading sharded files from S3/GCS.
# - **`collate_fn` is the key DataLoader concept for text.** Variable-length
#   sequences require either fixed-length padding in the Dataset or dynamic
#   padding via a custom `collate_fn`. The latter is more efficient for
#   production workloads.
# - **All three backends feed the same DataLoader.** Once texts and labels are
#   extracted as Python lists, the downstream pipeline (tokenisation, Dataset,
#   DataLoader, training) is identical.
