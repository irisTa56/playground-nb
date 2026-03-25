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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/daft/pytorch_dataloader_image.ipynb)
#
# # PyTorch DataLoader — Image Data
#
# Cat-vs-dog classification with **torchvision**, **Polars** (metadata), and **Daft** (multimodal).
#
# This notebook is **self-contained**: it installs its own dependencies, downloads
# the dataset, and detects the available accelerator — no prior setup is needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.10",
#   "torchvision>=0.22",
#   "polars>=1.39",
#   "daft>=0.7",
#   "Pillow>=10.0",
#   "matplotlib>=3.10",
#   "kaggle>=2.0",
#   "pandas>=3.0",
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
for pkg in ("torch", "torchvision", "polars", "daft", "Pillow", "matplotlib", "numpy"):
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

DATA_DIR = Path("data/catdog")
TRAIN_DIR = DATA_DIR / "training_set" / "training_set"
TEST_DIR = DATA_DIR / "test_set" / "test_set"

if not TRAIN_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "tongpython/cat-and-dog",
                "-p",
                str(DATA_DIR),
                "--unzip",
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Kaggle CLI failed ({exc}). Download manually from:")
        print("  https://www.kaggle.com/datasets/tongpython/cat-and-dog/data")
        print(f"and extract to {DATA_DIR}/")

    # Kaggle sometimes downloads as zip without --unzip working
    for zf in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(DATA_DIR, filter="data")  # type: ignore[call-arg]  # Python 3.12+
        zf.unlink()

n_train = sum(1 for _ in TRAIN_DIR.rglob("*.jpg"))
n_test = sum(1 for _ in TEST_DIR.rglob("*.jpg"))
print(f"Dataset ready: {n_train:,} train / {n_test:,} test images")

# %% [markdown]
# ## 1. `make_dataloader` Helper
#
# Platform-aware helper — `num_workers=0` on macOS to avoid multiprocessing
# issues, `pin_memory=True` when using CUDA. For image data, `num_workers`
# and `pin_memory` are the most impactful DataLoader parameters because image
# decoding and augmentation are CPU-bound.

# %%
from torch.utils.data import DataLoader, Dataset


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0 if sys.platform == "darwin" else (os.cpu_count() or 2) // 2,
    pin_memory: bool = DEVICE.type == "cuda",
    persistent_workers: bool | None = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for image tasks."""
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
# ## 2. `num_workers` and `pin_memory` — Tuning for Image I/O
#
# Image pipelines are typically **I/O and decode-bound**: each sample requires
# reading a JPEG from disk, decompressing it, and applying transforms (resize,
# normalize). Unlike tabular data (where tensors are already in memory) or text
# data (where the tokenizer dominates), image loading benefits significantly from
# parallel prefetching.
#
# | Parameter | Effect | Recommendation |
# |-----------|--------|----------------|
# | `num_workers` | Number of subprocesses to prefetch batches | Start with `(os.cpu_count() or 2) // 2`; set to 0 on macOS (multiprocessing issues) |
# | `pin_memory` | Allocates host tensors in page-locked memory | Set `True` when training on CUDA — speeds up CPU→GPU transfer |
# | `persistent_workers` | Keep worker processes alive between epochs | Set `True` when `num_workers > 0` to avoid respawn overhead |
# | `prefetch_factor` | Number of batches each worker prefetches | Default 2 is usually fine; increase if GPU is starved |

# %% [markdown]
# ---
# ## 3. Approach 1 — torchvision `ImageFolder` + `transforms.v2`
#
# The `torchvision.transforms.v2` API supersedes the legacy `transforms` module.
# It operates on tensors natively and integrates with `torch.compile`.
# `ImageFolder` reads images from a directory structure where each subdirectory
# is a class label.

# %%
from torchvision import datasets
from torchvision.transforms import v2 as T

transform = T.Compose(
    [
        T.Resize((224, 224), antialias=True),
        T.ToImage(),  # PIL Image → uint8 tensor
        T.ToDtype(torch.float32, scale=True),  # [0,255] → [0.0,1.0]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset_tv = datasets.ImageFolder(str(TRAIN_DIR), transform=transform)
test_dataset_tv = datasets.ImageFolder(str(TEST_DIR), transform=transform)

print(f"Classes: {train_dataset_tv.classes}")
print(f"Class→idx: {train_dataset_tv.class_to_idx}")
print(f"Train: {len(train_dataset_tv):,}  Test: {len(test_dataset_tv):,}")

# %%
train_loader_tv = make_dataloader(train_dataset_tv, batch_size=32, shuffle=True)
test_loader_tv = make_dataloader(test_dataset_tv, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader_tv))
print(f"Batch images shape: {images.shape}")  # [32, 3, 224, 224]
print(f"Batch labels shape: {labels.shape}")  # [32]
print(f"Label distribution: {labels.sum().item():.0f} dogs / {len(labels)} total")

# %% [markdown]
# ### Batch Visualisation

# %%
import matplotlib.pyplot as plt
import numpy as np


def show_batch(
    images: torch.Tensor, labels: torch.Tensor, class_names: list[str], n: int = 8
):
    """Display the first *n* images from a batch."""
    # Denormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imgs = images[:n] * std + mean
    imgs = imgs.clamp(0, 1)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(imgs[i].numpy(), (1, 2, 0)))
        ax.set_title(class_names[int(labels[i].item())])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


show_batch(images, labels, train_dataset_tv.classes)

# %% [markdown]
# ---
# ## 4. Approach 2 — Polars Metadata + Custom Dataset
#
# Polars is **not designed to hold pixel data** as columns. It is useful for
# managing image path metadata — building file lists, joining labels, filtering
# by attributes — while image decoding stays on the torchvision side.

# %%
import polars as pl

# Build a metadata DataFrame with expression-based label extraction
meta_train = pl.DataFrame(
    {"path": [str(p) for p in TRAIN_DIR.rglob("*.jpg")]}
).with_columns(pl.col("path").str.contains("dogs").cast(pl.Int8).alias("label"))

meta_test = pl.DataFrame(
    {"path": [str(p) for p in TEST_DIR.rglob("*.jpg")]}
).with_columns(pl.col("path").str.contains("dogs").cast(pl.Int8).alias("label"))

print(f"Train metadata: {len(meta_train):,} rows")
print(
    f"  cats: {len(meta_train) - meta_train['label'].sum()}, dogs: {meta_train['label'].sum()}"
)
meta_train.head()

# %%
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class ImagePathDataset(TorchDataset):
    """Map-style dataset that loads images from file paths."""

    def __init__(self, paths: list[str], labels: list[int], transform=None) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):  # type: ignore[override]
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


train_dataset_pl = ImagePathDataset(
    meta_train["path"].to_list(),
    meta_train["label"].to_list(),
    transform=transform,
)
test_dataset_pl = ImagePathDataset(
    meta_test["path"].to_list(),
    meta_test["label"].to_list(),
    transform=transform,
)

print(
    f"Polars datasets — Train: {len(train_dataset_pl):,}  Test: {len(test_dataset_pl):,}"
)

# Verify a sample
img, label = train_dataset_pl[0]
print(f"Sample shape: {img.shape}, label: {label}")

# %% [markdown]
# ---
# ## 5. Approach 3 — Daft Multimodal Pipeline
#
# Daft has **native support for image columns**. It can discover files, extract
# labels, download image bytes, and decode — all within a single DataFrame.
# Cloud paths work transparently: `daft.from_glob_path("s3://bucket/images/*")`
# requires no additional configuration.

# %%
import daft
from daft import DataType, col

# Suppress Daft's tqdm progress bar
os.environ["DAFT_PROGRESS_BAR"] = "0"

train_df = daft.from_glob_path(str(TRAIN_DIR / "*" / "*.jpg"))

# Expression-based label extraction (no Python loop over rows)
train_df = train_df.with_column(
    "label",
    col("path").contains("dogs").cast(DataType.int8()),
)

train_df_collected = train_df.select("path", "label").collect()
print(f"Daft train DataFrame: {len(train_df_collected):,} rows")
train_df_collected.to_pandas().head()

# %%
test_df = daft.from_glob_path(str(TEST_DIR / "*" / "*.jpg"))
test_df = test_df.with_column(
    "label",
    col("path").contains("dogs").cast(DataType.int8()),
)
test_df_collected = test_df.select("path", "label").collect()
print(f"Daft test DataFrame: {len(test_df_collected):,} rows")

# %% [markdown]
# ### Daft → `IterableDataset` → `DataLoader`
#
# Convert a Daft DataFrame to an `IterableDataset` via `to_torch_iter_dataset()`.
# Each row yields a dict; we wrap it in a custom `IterableDataset` to apply
# torchvision transforms and yield `(image_tensor, label)` tuples.

# %%
from torch.utils.data import IterableDataset


class DaftPathStream(IterableDataset):
    """Stream images from a Daft DataFrame via file paths."""

    def __init__(self, daft_df: daft.DataFrame, transform=None) -> None:
        self.daft_df = daft_df
        self.transform = transform or (lambda x: x)

    def __iter__(self):
        for row in self.daft_df.select("path", "label").to_torch_iter_dataset():
            img_path = Path(row["path"].replace("file://", ""))
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            yield img, row["label"]


train_dataset_daft = DaftPathStream(train_df, transform=transform)
test_dataset_daft = DaftPathStream(test_df, transform=transform)

# IterableDataset does not support shuffle — Daft handles row ordering internally
train_loader_daft = make_dataloader(train_dataset_daft, batch_size=32, shuffle=False)

# Verify a batch
batch_imgs, batch_lbls = next(iter(train_loader_daft))
print(f"Daft batch images: {batch_imgs.shape}, labels: {batch_lbls.shape}")

# %% [markdown]
# ---
# ## 6. Training & Evaluation
#
# A lightweight CNN classifier trained for a few epochs on a small subset to
# verify the end-to-end pipeline. We compare torchvision `ImageFolder` and
# the Polars metadata approach (both are map-style Datasets).
#
# > **Note:** The Daft `IterableDataset` approach is also valid but is skipped
# > in the timed comparison since `IterableDataset` does not support `len()`,
# > making epoch-based training less straightforward.

# %%
import timeit

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Minimal CNN for binary image classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 → 112
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 → 56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 56 → 1
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(-1)


def train_and_evaluate(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    num_epochs: int = 3,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Run train → eval pipeline and return metrics."""
    torch.manual_seed(seed)

    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # --- Training ---
    train_loss = 0.0
    train_acc = 0.0
    for _epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in train_loader:
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE, dtype=torch.float32)
            logits = model(imgs)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += ((logits > 0).float() == lbls).sum().item()
            total += imgs.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

    # --- Evaluation ---
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE, dtype=torch.float32)
            logits = model(imgs)
            test_loss += criterion(logits, lbls).item() * imgs.size(0)
            correct += ((logits > 0).float() == lbls).sum().item()
            total += imgs.size(0)

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss / total,
        "test_acc": correct / total,
    }


# %%
import pandas as pd

# Use a subset for faster training
MAX_TRAIN = 2000
MAX_TEST = 500

# Subset via torch.utils.data.Subset
from torch.utils.data import Subset

torch.manual_seed(42)
train_indices = torch.randperm(len(train_dataset_tv))[:MAX_TRAIN].tolist()
test_indices = torch.randperm(len(test_dataset_tv))[:MAX_TEST].tolist()

approaches: dict[str, tuple[DataLoader, DataLoader]] = {
    "torchvision": (
        make_dataloader(
            Subset(train_dataset_tv, train_indices), batch_size=32, shuffle=True
        ),
        make_dataloader(
            Subset(test_dataset_tv, test_indices), batch_size=32, shuffle=False
        ),
    ),
    "Polars": (
        make_dataloader(
            Subset(train_dataset_pl, train_indices), batch_size=32, shuffle=True
        ),
        make_dataloader(
            Subset(test_dataset_pl, test_indices), batch_size=32, shuffle=False
        ),
    ),
}

# Warmup: absorb one-time costs (MPS shader compilation, etc.)
_warmup_tr, _warmup_te = next(iter(approaches.values()))
_ = train_and_evaluate(_warmup_tr, _warmup_te, num_epochs=1)
print("Warmup done.\n")

results: dict[str, dict[str, float]] = {}
for name, (tr_loader, te_loader) in approaches.items():
    t0 = timeit.default_timer()
    metrics = train_and_evaluate(tr_loader, te_loader)
    elapsed = timeit.default_timer() - t0
    results[name] = {**metrics, "time_sec": elapsed}
    print(
        f"{name:>12}  train_loss={metrics['train_loss']:.4f}  "
        f"train_acc={metrics['train_acc']:.3f}  "
        f"test_loss={metrics['test_loss']:.4f}  "
        f"test_acc={metrics['test_acc']:.3f}  "
        f"time={elapsed:.1f}s"
    )

# %% [markdown]
# Both approaches produce comparable results — the `ImageFolder` and Polars
# metadata pipelines feed the **same images and labels** to the DataLoader.
# The difference is in how paths and labels are discovered and managed.

# %%
results_df = pd.DataFrame(results).T
results_df.index.name = "approach"
results_df

# %% [markdown]
# ---
# ## 7. Image Pipeline Comparison
#
# | Aspect | torchvision `ImageFolder` | Polars metadata | Daft multimodal |
# |--------|--------------------------|-----------------|-----------------|
# | Native image column | — | No | **Yes** |
# | Decode / resize | `transforms.v2` | `transforms.v2` (via custom Dataset) | Built-in + `transforms.v2` usable together |
# | Cloud image loading | Custom Dataset impl. | Custom Dataset impl. | `url.download()` built-in |
# | Metadata joins | Not supported | SQL-like joins | DataFrame join + lazy optimisation |
# | Dataset type | Map-style | Map-style | `IterableDataset` (streaming) |
# | Simplicity | ★★★★★ (one-liner) | ★★★★☆ (flexible) | ★★★☆☆ (powerful but more setup) |
#
# ### Key Observations
#
# - **`num_workers` matters most for images.** Image decoding is CPU-bound;
#   parallel workers prevent the GPU from starving. This is the DataLoader
#   concept most relevant to image pipelines.
# - **`pin_memory` speeds up CUDA transfers.** Page-locked host memory enables
#   async DMA copies to the GPU, which is especially impactful for large
#   image batches.
# - **torchvision `ImageFolder` is the simplest approach.** For local datasets
#   with a directory-per-class structure, it requires minimal code.
# - **Polars excels at metadata management.** When you need to join labels from
#   a CSV, filter by attributes, or sample images, Polars provides a fast and
#   expressive API — but leaves decoding to torchvision.
# - **Daft enables end-to-end multimodal pipelines.** Native image columns,
#   built-in cloud I/O, and lazy execution make Daft uniquely suited for
#   large-scale image datasets, especially on cloud storage.
