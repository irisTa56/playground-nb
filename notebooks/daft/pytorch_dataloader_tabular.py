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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/daft/pytorch_dataloader_tabular.ipynb)
#
# # PyTorch DataLoader — Tabular Data
#
# House-price prediction with **Pandas**, **Polars**, and **Daft**.
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
#   "getdaft>=0.5",
#   "scikit-learn>=1.8",
#   "matplotlib>=3.10",
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
for pkg in ("torch", "pandas", "polars", "daft", "scikit-learn", "numpy"):
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

DATA_DIR = Path("data/housing")
CSV_PATH = DATA_DIR / "Housing.csv"

if not CSV_PATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "yasserh/housing-prices-dataset",
                "-p",
                str(DATA_DIR),
                "--unzip",
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Kaggle CLI failed ({exc}). Download manually from:")
        print("  https://www.kaggle.com/datasets/yasserh/housing-prices-dataset")
        print(f"and extract Housing.csv to {DATA_DIR}/")

    # Kaggle sometimes downloads as zip without --unzip working
    for zf in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(DATA_DIR, filter="data")  # type: ignore[call-arg]  # Python 3.12+
        zf.unlink()

print(f"Dataset ready: {CSV_PATH}  ({CSV_PATH.stat().st_size:,} bytes)")

# %% [markdown]
# ## 1. Dataset ↔ DataLoader Relationship
#
# **`Dataset`** defines *what* each sample looks like (via `__getitem__` and `__len__`).
#
# **`DataLoader`** wraps a Dataset and handles *how* samples are delivered to the
# training loop:
#
# - **Automatic batching** — collects `batch_size` samples into tensors
# - **Shuffling** — `shuffle=True` randomises order each epoch
# - **Parallel loading** — `num_workers` subprocesses prefetch data
# - **Custom collation** — `collate_fn` for variable-length sequences, padding, etc.
# - **Pin memory** — `pin_memory=True` accelerates CPU → GPU transfer
#
# ```
# Dataset ──▶ DataLoader ──▶ Training Loop
# (one sample)  (batches)     (model.forward)
# ```

# %%
from torch.utils.data import DataLoader, TensorDataset

X_demo = torch.randn(100, 3)  # 100 samples, 3 features
y_demo = torch.randn(100, 1)

demo_dataset = TensorDataset(X_demo, y_demo)
demo_loader = DataLoader(demo_dataset, batch_size=16, shuffle=True)

batch = next(iter(demo_loader))
print(f"Batch features shape: {batch[0].shape}")  # [16, 3]
print(f"Batch labels  shape: {batch[1].shape}")  # [16, 1]

# %% [markdown]
# ## 2. Core DataLoader Parameters
#
# | Parameter | Purpose | Typical value |
# |-----------|---------|---------------|
# | `batch_size` | Samples per batch | 32–256 (tabular), 16–64 (vision/NLP) |
# | `shuffle` | Randomise order each epoch | `True` for training, `False` for eval |
# | `num_workers` | Parallel data-loading subprocesses | 0–4 (see macOS note below) |
# | `pin_memory` | Page-locked memory for faster GPU transfer | `True` when using GPU |
# | `collate_fn` | Custom batching logic | Default works for uniform tensors |
# | `persistent_workers` | Keep workers alive between epochs | `True` to amortise spawn cost |

# %%
from torch.utils.data import Dataset


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0 if sys.platform == "darwin" else (os.cpu_count() or 2) // 2,
    pin_memory: bool = DEVICE.type == "cuda",
    persistent_workers: bool | None = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with sensible defaults."""
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs,
    )


# %% [markdown]
# ---
# ## 3. Preprocessing with Pandas, Polars, and Daft
#
# We load the same Housing dataset with all three libraries, encode categorical
# columns, and convert to NumPy arrays for the shared `HousePriceDataset`.

# %% [markdown]
# ### 3.1 Pandas (Traditional Approach)
#
# Pandas uses **eager evaluation** — `pd.read_csv()` reads the entire file into
# memory immediately. This is fine for small datasets but risks OOM errors beyond
# a few GB.

# %%
import timeit

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/test and StandardScaler-transform both X and y."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train).astype(np.float32)
    X_test = scaler_X.transform(X_test).astype(np.float32)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)
    return X_train, X_test, y_train, y_test


df_pd = pd.read_csv(CSV_PATH)
print(f"Pandas shape: {df_pd.shape}")
df_pd.head()

# %%
# Encode binary yes/no columns as integers
binary_cols = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
df_pd[binary_cols] = (df_pd[binary_cols] == "yes").astype(np.int8)

# Encode multi-class column with explicit mapping (deterministic & reversible)
furnishing_order = ["unfurnished", "semi-furnished", "furnished"]  # 0, 1, 2
df_pd["furnishingstatus"] = pd.Categorical(
    df_pd["furnishingstatus"], categories=furnishing_order, ordered=True
).codes.astype(np.int8)

# Convert to NumPy first — avoids pandas 3.0 Copy-on-Write pitfalls
X_pd_np = df_pd.drop(columns=["price"]).to_numpy(dtype=np.float32)
y_pd = df_pd["price"].to_numpy(dtype=np.float32)

X_train_pd_np, X_test_pd_np, y_train_pd, y_test_pd = split_and_scale(X_pd_np, y_pd)

print(f"Pandas — train: {X_train_pd_np.shape}, test: {X_test_pd_np.shape}")

# %% [markdown]
# ### 3.2 Polars (Zero-Copy Arrow-Based)
#
# Polars uses **Apache Arrow** as its in-memory format (columnar, cache-friendly).
# The Rust implementation avoids Python's GIL entirely.

# %%
import polars as pl

df_pl = pl.read_csv(CSV_PATH)
print(f"Polars shape: {df_pl.shape}")

# Binary columns: vectorised replace + cast
df_pl = df_pl.with_columns(
    pl.col(col).replace_strict({"yes": 1, "no": 0}).cast(pl.Int8) for col in binary_cols
)

# Multi-class: pl.Enum encodes ordering semantically, to_physical() is vectorised
furnishing_order = ["unfurnished", "semi-furnished", "furnished"]
df_pl = df_pl.with_columns(
    pl.col("furnishingstatus")
    .cast(pl.Enum(furnishing_order))
    .to_physical()
    .cast(pl.Int8)
    .alias("furnishingstatus")
)

X_pl_np = df_pl.drop("price").to_numpy().astype(np.float32)
y_pl = df_pl["price"].to_numpy().astype(np.float32)

X_pl_train, X_pl_test, y_pl_train, y_pl_test = split_and_scale(X_pl_np, y_pl)

print(f"Polars — train: {X_pl_train.shape}, test: {X_pl_test.shape}")

# %%
# Polars Lazy API: scan_csv enables predicate pushdown and projection pushdown —
# only the rows and columns actually needed are read from disk.
df_lazy: pl.DataFrame = (
    pl.scan_csv(CSV_PATH)
    .select(["area", "bedrooms", "price"])
    .filter(pl.col("price") > 100_000)
    .collect()
)  # type: ignore[assignment]
print(f"Polars Lazy (area, bedrooms where price > 100k): {df_lazy.shape}")
df_lazy.head()

# %% [markdown]
# ### 3.3 Daft (Cloud-Native + Multimodal)
#
# Daft uses **lazy evaluation by default** — nothing executes until `.collect()`.
# Its Rust query engine supports native S3/GCS/Azure access.

# %%
import daft
from daft import DataType, col

# Suppress Daft's tqdm progress bar (e.g. "Read CSV: 50,000 rows out")
os.environ["DAFT_PROGRESS_BAR"] = "0"

df_daft = daft.read_csv(str(CSV_PATH))

# Expression-based binary encoding (no Python loops over rows)
df_daft = df_daft.with_columns(
    {c: col(c).contains("yes").cast(DataType.int8()) for c in binary_cols}
)

# Multi-class: boolean arithmetic avoids Python UDF (Daft 0.7 lacks if_else)
# furnished → 2, semi-furnished → 1, unfurnished → 0
is_furnished = col("furnishingstatus").eq_null_safe("furnished").cast(DataType.int8())
is_semi_furnished = (
    col("furnishingstatus").eq_null_safe("semi-furnished").cast(DataType.int8())
)
is_unfurnished = (
    col("furnishingstatus").eq_null_safe("unfurnished").cast(DataType.int8())
)

df_daft = df_daft.with_columns(
    {
        "furnishingstatus": is_furnished * 2 + is_semi_furnished,
        "_matched": is_furnished + is_semi_furnished + is_unfurnished,
    }
)

# Lazy execution: trigger with .collect()
df_daft = df_daft.collect()

# Validate & drop helper column via pydict (no extra Daft scans)
data = df_daft.to_pydict()
assert min(data.pop("_matched")) == 1, "Unknown furnishingstatus values found"
df_daft = daft.from_pydict(data).collect()  # type: ignore[arg-type]

print(f"Daft — rows: {len(df_daft)}")
df_daft.to_pandas().head()

# %%
# Daft lacks a direct to_numpy(); convert via Arrow → Pandas → NumPy
df_daft_pd = df_daft.to_pandas()
X_daft_np = df_daft_pd.drop(columns=["price"]).to_numpy(dtype=np.float32)
y_daft = df_daft_pd["price"].to_numpy(dtype=np.float32)

X_daft_train, X_daft_test, y_daft_train, y_daft_test = split_and_scale(
    X_daft_np, y_daft
)

print(f"Daft — train: {X_daft_train.shape}, test: {X_daft_test.shape}")

# %% [markdown]
# ---
# ## 4. Dataset → DataLoader → Training (All Backends)
#
# Each library produced the same shape of NumPy arrays. We now feed all three
# into **identical** DataLoader + training pipelines and compare the results.
# This confirms that the preprocessing backend is interchangeable — the
# DataLoader sees the same tensors regardless of how they were prepared.

# %%
from torch.utils.data import Dataset as TorchDataset


class HousePriceDataset(TorchDataset):
    """Map-style dataset for house-price regression."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.X = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.X[idx], self.y[idx]


class SimpleRegressor(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Run full DataLoader → train → eval pipeline and return metrics."""
    torch.manual_seed(seed)

    train_ds = HousePriceDataset(X_train, y_train)
    test_ds = HousePriceDataset(X_test, y_test)
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = make_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    model = SimpleRegressor(in_features=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- Training ---
    for _epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(X_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)
        epoch_loss /= len(train_ds)

    # --- Evaluation ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(X_b)
            test_loss += criterion(pred, y_b).item() * X_b.size(0)
    test_loss /= len(test_ds)

    return {"final_train_loss": epoch_loss, "test_loss": test_loss}


# %%
backends: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
    "Pandas": (X_train_pd_np, X_test_pd_np, y_train_pd, y_test_pd),
    "Polars": (X_pl_train, X_pl_test, y_pl_train, y_pl_test),
    "Daft": (X_daft_train, X_daft_test, y_daft_train, y_daft_test),
}

N_RUNS = 10

# Warmup: absorb one-time costs (PyTorch lazy init, MPS shader compilation, etc.)
_warmup_data = next(iter(backends.values()))
_ = train_and_evaluate(*_warmup_data)
print("Warmup done.\n")

results: dict[str, dict[str, float]] = {}
for name, (Xtr, Xte, ytr, yte) in backends.items():
    times: list[float] = []
    metrics: dict[str, float] = {}
    for _ in range(N_RUNS):
        t0 = timeit.default_timer()
        metrics = train_and_evaluate(Xtr, Xte, ytr, yte)
        times.append(timeit.default_timer() - t0)

    mean_t = np.mean(times)
    std_t = np.std(times)
    results[name] = {
        "final_train_loss": metrics["final_train_loss"],
        "test_loss": metrics["test_loss"],
        "time_mean_sec": float(mean_t),
        "time_std_sec": float(std_t),
    }
    print(
        f"{name:>6}  train_loss={metrics['final_train_loss']:.4f}  "
        f"test_loss={metrics['test_loss']:.4f}  "
        f"time={mean_t:.3f}s ± {std_t:.3f}s  ({N_RUNS} runs)"
    )

# %% [markdown]
# All three backends produce **nearly identical** losses — small differences stem
# from floating-point ordering in the encoding step, not from the DataLoader
# itself. This confirms the key point: **the DataLoader is agnostic to the
# preprocessing backend**.

# %%
results_df = pd.DataFrame(results).T
results_df.index.name = "backend"
results_df

# %% [markdown]
# ## 5. Tabular Preprocessing Comparison
#
# ### Library Characteristics
#
# | Aspect | Pandas | Polars | Daft |
# |--------|--------|--------|------|
# | Evaluation model | Eager (immediate) | Eager + **Lazy API** | **Lazy** (default) |
# | Memory format | NumPy-based (row-oriented) | **Apache Arrow** (columnar) | **Apache Arrow** (columnar) |
# | GIL impact | Yes (Python impl.) | **None** (Rust impl.) | **None** (Rust impl.) |
# | Column pruning | None (all columns loaded) | Automatic in Lazy mode | Automatic in Lazy mode |
# | NumPy conversion cost | Copy incurred | Near-zero-copy | Via Arrow → Pandas → NumPy |
# | Cloud I/O | Requires fsspec / boto3 | Partial S3 support | **Native S3/GCS/Azure** |
# | Ecosystem maturity | ★★★★★ | ★★★★☆ | ★★★☆☆ |
#
# ### What We Observed
#
# - **Test losses are nearly identical** across all three backends — the
#   DataLoader receives the same tensors regardless of which library prepared them.
# - Differences in loss (if any) come from floating-point ordering during
#   categorical encoding, not from the DataLoader or training loop.
# - The `results_df` table above shows the actual pipeline timing. On this
#   small dataset the differences are negligible; on larger datasets, Polars
#   and Daft's lazy evaluation and columnar format provide real advantages.
#
# **Key takeaway:** The preprocessing library affects *how efficiently* data
# reaches NumPy, but once it arrives, **DataLoader behaviour is identical**.
