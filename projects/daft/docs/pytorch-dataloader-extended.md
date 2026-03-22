# PyTorch DataLoader and DataFrame Library Selection: Pandas / Polars / Daft

> Based on the original tutorial at [daft.ai/blog/pytorch-data-loader](https://www.daft.ai/blog/pytorch-data-loader), extended with a natural comparison of Pandas, Polars, and Daft at the preprocessing stage.

---

## Introduction

PyTorch's `DataLoader` is the core utility responsible for feeding data to model training.
It handles batching, shuffling, and multi-worker parallel loading — everything the training loop needs, in one place.

However, **the data preprocessing that happens before passing data to the DataLoader** is handled by a variety of DataFrame libraries.
Pandas has long been the default, but in recent years Polars and Daft have emerged as compelling alternatives.
This tutorial walks through practical examples across modalities (tabular, text, image) to explain how to use the PyTorch DataLoader, while comparing how the choice of preprocessing library affects the overall pipeline.

---

## 1. PyTorch DataLoader Fundamentals

`DataLoader` is a class in `torch.utils.data` that wraps a `Dataset` object and produces an iterator of batches.
Its main features are:

- **Automatic batching**: collects `batch_size` samples into tensors
- **Shuffling**: `shuffle=True` randomises data order each epoch
- **Parallel loading**: `num_workers` enables subprocess-based parallelism
- **Custom collation**: `collate_fn` handles variable-length sequences, padding, etc.
- **Pin memory**: `pin_memory=True` accelerates CPU-to-GPU transfer by keeping tensors in page-locked memory

```python
for batch in train_loader:
    out = model(batch[0])
    loss = compute_loss(out, batch[1])
    ...
```

The DataLoader is agnostic about the content of the data — it only batches and delivers the samples returned by the Dataset.
This means **which library performed preprocessing has no bearing on DataLoader behaviour**.
What it does affect is the throughput and memory efficiency of the preprocessing stage itself.

> **`num_workers` note**: On Linux, `num_workers=4` (or `os.cpu_count() // 2`) is a reasonable starting point.
> On macOS/Windows, multi-process workers have higher overhead; keep `num_workers` low (0–2) or use `persistent_workers=True` to amortise the spawn cost.

---

## 2. Tabular Data: Where the Choice of Preprocessing Library Matters Most

Using a house-price prediction task ([Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)) as an example, we implement the same preprocessing in all three libraries.

### 2.1 Pandas (Traditional Approach)

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Housing.csv")

# Encode binary yes/no columns as integers
binary_cols = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea",
]
dataset[binary_cols] = dataset[binary_cols].apply(
    lambda s: s.map({"yes": 1, "no": 0})
)

# Encode multi-class column with explicit mapping to avoid non-deterministic ordering
furnishing_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
dataset["furnishingstatus"] = dataset["furnishingstatus"].map(furnishing_map)

X = dataset.drop(columns=["price"])
y = dataset["price"].to_numpy(dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardise numeric columns
numeric_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

X_train_np = X_train.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)
```

Pandas loads the entire dataset into memory at `pd.read_csv` time (**eager evaluation**).
This is fine for small-to-medium data, but risks OOM errors beyond a few GB.
Using explicit `map` dictionaries rather than `LabelEncoder` makes the encoding deterministic and reversible.

### 2.2 Polars (Zero-Copy Arrow-Based)

```python
from __future__ import annotations

import numpy as np
import polars as pl

dataset = pl.read_csv("Housing.csv")

# Binary columns: cast "yes"/"no" strings directly to Int8 using a map
binary_cols = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea",
]
dataset = dataset.with_columns(
    pl.col(col).replace_strict({"yes": 1, "no": 0}).cast(pl.Int8)
    for col in binary_cols
)

# Multi-class column: use pl.Enum for memory-efficient, type-safe encoding
furnishing_order = ["unfurnished", "semi-furnished", "furnished"]
dataset = dataset.with_columns(
    pl.col("furnishingstatus")
    .cast(pl.Enum(furnishing_order))
    .to_physical()           # converts Enum ordinal to UInt32
    .cast(pl.Int8)
    .alias("furnishingstatus")
)

X = dataset.drop("price")
y = dataset["price"].to_numpy(dtype=np.float32)

X_np = X.to_numpy().astype(np.float32)
```

Polars uses Apache Arrow as its in-memory format, giving it a column-oriented, cache-friendly layout.
Using `pl.Enum` for categorical columns is preferred over manual integer loops because it encodes the ordering semantically, avoids Python-level iteration, and the `to_physical()` expression is fully vectorised.

The **Lazy API** (`pl.scan_csv`) additionally enables predicate and projection pushdown, so only the rows and columns you actually need are read from disk:

```python
# Lazy mode: optimisation happens automatically before collect()
dataset = (
    pl.scan_csv("Housing.csv")
    .select(["area", "bedrooms", "price"])  # projection pushdown
    .filter(pl.col("price") > 100_000)      # predicate pushdown
    .collect()
)
```

### 2.3 Daft (Cloud-Native + Multimodal)

```python
from __future__ import annotations

import daft
from daft import col, DataType

dataset = daft.read_csv("Housing.csv")

# Expression-based binary encoding (no Python loops over rows)
binary_cols = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea",
]
dataset = dataset.with_columns(
    {c: col(c).is_in(["yes"]).cast(DataType.int8()) for c in binary_cols}
)

furnishing_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
dataset = dataset.with_column(
    "furnishingstatus",
    col("furnishingstatus").apply(
        lambda x: furnishing_map[x], return_dtype=DataType.int8()
    ),
)

# Daft is lazy by default — nothing runs until .collect()
dataset = dataset.collect()
```

Daft uses a Rust query engine with lazy evaluation as its default mode.
For local files the performance is comparable to Polars, but its biggest differentiator is **native access to S3, GCS, and Azure Blob Storage** without any additional driver code.

### 2.4 Shared Code After Preprocessing: Dataset → DataLoader

Regardless of which library performed preprocessing, once the data is in a NumPy array the path to `DataLoader` is identical.

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class HousePriceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        # as_tensor avoids a copy when the array is already float32 and C-contiguous
        self.X = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


train_dataset = HousePriceDataset(X_train_np, y_train_norm)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,   # keeps tensors in page-locked memory for faster GPU transfer
    persistent_workers=True,  # avoids re-spawning workers each epoch
)
```

Key points:

- `torch.as_tensor()` is preferred over `torch.tensor()` for NumPy arrays because it **avoids a memory copy** when the dtype and memory layout already match.
  Use `torch.tensor()` only when you want an unconditional copy or are creating a tensor from Python scalars/lists.
- `.unsqueeze(1)` is cleaner than `.view(-1, 1)` when reshaping a 1-D label vector.

**The choice of preprocessing library does not affect DataLoader behaviour.**
It only affects the speed, memory efficiency, and cloud compatibility of the preprocessing phase.

### 2.5 Tabular Preprocessing Comparison Summary

| Aspect | Pandas | Polars | Daft |
|--------|--------|--------|------|
| Evaluation model | Eager (immediate) | Eager + **Lazy API** | **Lazy** (default) |
| Memory format | NumPy-based (row-oriented) | **Apache Arrow** (columnar) | **Apache Arrow** (columnar) |
| GIL impact | Yes (Python impl.) | **None** (Rust impl.) | **None** (Rust impl.) |
| Column pruning | None (all columns loaded) | Automatic in Lazy mode | Automatic in Lazy mode |
| NumPy conversion cost | Copy incurred | Near-zero-copy | Via Arrow |
| Cloud I/O | Requires fsspec / boto3 | Partial S3 support | **Native S3/GCS/Azure** |
| Ecosystem maturity | ★★★★★ | ★★★★☆ | ★★★☆☆ |

---

## 3. Text Data: BERT + IMDB Sentiment Classification

For text preprocessing, the tokenizer is the dominant step, so the choice of DataFrame library has less impact on overall speed than it does for tabular data.
The difference becomes relevant at the data-loading stage for large datasets.

### 3.1 Data Loading Comparison

```python
# --- Pandas ---
dataset = pd.read_csv("imdb_test.csv")[["text", "label"]].dropna()

# --- Polars (fast Rust-based CSV parser) ---
dataset = pl.read_csv("imdb_test.csv").select(["text", "label"]).drop_nulls()

# --- Daft (lazy evaluation; strong for large or remote files) ---
dataset = (
    daft.read_csv("imdb_test.csv")
    .select("text", "label")
    .collect()
)
```

For CSVs of several hundred MB or more, Polars' parser substantially outperforms Pandas in most benchmarks.
Daft may be slower than Polars for a single local CSV, but benefits from distributed I/O when reading sharded files from S3.

### 3.2 Dataset and DataLoader (Shared)

Once loaded, the downstream code is independent of the DataFrame library used.

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class IMDBDataset(Dataset):
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return input_ids, attention_mask, label


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
```

---

## 4. Image Data: Where Library Differences Are Most Pronounced

Using a cat-vs-dog classification task ([Kaggle Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)), we examine library differences in image pipelines.

### 4.1 Modern Approach: torchvision.transforms.v2 + DataLoader

The `torchvision.transforms.v2` API (introduced in torchvision 0.15) supersedes the legacy `transforms` module.
It operates on tensors natively, supports `torchscript`, and integrates with `torch.compile`.
Prefer it for all new projects.

```python
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T

transform = T.Compose([
    T.Resize((224, 224), antialias=True),
    T.ToImage(),                          # converts PIL Image → uint8 tensor
    T.ToDtype(torch.float32, scale=True), # normalises [0,255] → [0.0,1.0]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(
    Path("training_set/"), transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

`ImageFolder` is a map-style Dataset that reads, decodes, and transforms each image from disk on demand in `__getitem__`.
It works well for local data, but requires a custom Dataset when images live on S3.

### 4.2 The Role of Pandas / Polars

Pandas and Polars are **not designed to hold pixel data as columns**.
They are useful for managing image path metadata, but pixel-level operations are outside their scope.

```python
from __future__ import annotations

from pathlib import Path

import polars as pl

# Build a metadata DataFrame with expression-based label extraction
image_root = Path("training_set")
meta = pl.DataFrame(
    {"path": [str(p) for p in image_root.rglob("*.jpg")]}
).with_columns(
    pl.col("path").str.contains("cats").cast(pl.Int8).alias("label")
)

# Pass meta["path"].to_list() and meta["label"].to_list() to a custom Dataset
```

Image decoding and resizing still happen on the torchvision side; Polars only manages the metadata table.

### 4.3 Daft: Multimodal Pipelines Within a Single DataFrame

Daft has native support for image and tensor columns.

```python
from __future__ import annotations

import daft
from daft import DataType, col

train_df = daft.from_glob_path("training_set/training_set/*/*.*")

# Expression-based label extraction (no Python loop over rows)
train_df = train_df.with_column(
    "label",
    col("path").str.contains("cats").cast(DataType.int8()),
)

# Download image bytes and decode in-engine
train_df = train_df.with_column(
    "image_bytes", col("path").url.download(on_error="null")
)
train_df = train_df.with_column(
    "image", col("image_bytes").image.decode()
)
```

Being able to view, resize, and filter images directly inside a DataFrame is a capability unique to Daft among the three libraries.
Cloud paths work transparently: `daft.from_glob_path("s3://bucket/images/*")` requires no additional configuration.

### 4.4 Daft → PyTorch DataLoader

Convert a Daft DataFrame to an `IterableDataset` and pass it to `DataLoader`.

```python
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import v2 as T

transform = T.Compose([
    T.Resize((224, 224), antialias=True),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DaftPathStream(IterableDataset):
    def __init__(self, datapipe, transform=None) -> None:
        self.datapipe = datapipe
        self.transform = transform or (lambda x: x)

    def __iter__(self):
        for row in self.datapipe:
            img_path = Path(row["path"].replace("file://", ""))
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            yield img, row["label"]


train_iter = train_df.select("path", "label").to_torch_iter_dataset()
train_dataset = DaftPathStream(train_iter, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
```

### 4.5 Image Pipeline Comparison

| Aspect | Pandas / Polars | torchvision direct | Daft |
|--------|-----------------|--------------------|------|
| Native image column | No | — | Yes |
| Decode / resize | External library required | `transforms.v2` | Built-in + `transforms.v2` usable together |
| Cloud image loading | fsspec + custom code | Custom Dataset impl. | `url.download()` built-in |
| Metadata joins | SQL-like joins | Custom in Dataset | DataFrame join + lazy optimisation |
| Streaming | No | `IterableDataset` | Auto-converts to `IterableDataset` |

---

## 5. Limitations of the Standard DataLoader and How Each Library Helps

### 5.1 Common Pain Points

- **I/O bottleneck**: at millions of samples, the Python-based DataLoader can become saturated
- **Worker memory duplication**: with `num_workers > 0`, the Dataset object is copied into each worker process
- **No native cloud storage**: accessing S3/GCS requires custom implementation
- **No distributed coordination**: data sharding for multi-node training requires additional tooling

### 5.2 What Each Library Covers

| Pain Point | Pandas | Polars | Daft |
|------------|--------|--------|------|
| Fast large CSV/Parquet ingestion | Slow | Rust parser | Rust + distributed |
| Memory efficiency | Many copies | Arrow zero-copy | Arrow + Lazy |
| Lazy evaluation to skip unused data | No | `scan_*` API | Default behaviour |
| Cloud storage integration | Requires fsspec | Partial S3 | Native S3/GCS/Azure |
| Multimodal type columns | No | No | Images, tensors, URLs |
| Distributed execution | No | Single machine only | Ray-based distribution |
| PyTorch integration | `.to_numpy()` → tensor | `.to_numpy()` → tensor | `.to_torch_iter_dataset()` |
| Ecosystem / documentation | Excellent | Good | Growing |

---

## 6. Decision Guide

### Kaggle / Research Prototype (up to a few GB, local)

Pandas or Polars + torchvision is recommended.
If you already have Pandas code, Pandas is perfectly sufficient.
For new projects where raw speed matters, try Polars' `scan_csv` / `scan_parquet`.

### Mid-Scale Production (tens of GB, local or cloud)

Polars (preprocessing) + DataLoader is recommended.
Polars' Lazy API with predicate and projection pushdown, combined with multi-threaded Rust execution, directly accelerates the preprocessing stage.
For cloud data, `pl.scan_parquet("s3://...")` works out of the box.

### Large-Scale Multimodal (TB-scale, cloud, distributed training)

Daft is recommended.
Managing images, text, and metadata in a single DataFrame, with the query engine handling cloud I/O optimisation and distributed execution, is Daft's unique advantage.
`to_torch_iter_dataset()` makes PyTorch integration straightforward.

---

## 7. Summary

PyTorch's DataLoader is the foundational tool for supplying data to model training via a consistent interface, regardless of modality.
However, **which library you use to build the preprocessing pipeline before the DataLoader** significantly affects the pipeline's throughput, memory efficiency, and cloud readiness.

- **Pandas**: the most mature ecosystem.
  Best when working with small-to-medium data or when existing code compatibility matters.
- **Polars**: fast Rust-based parser and Lazy API.
  Best when you want to accelerate preprocessing for local or mid-scale cloud data.
- **Daft**: multimodal type columns, native cloud I/O, distributed execution.
  Best when you want to unify a large-scale ML pipeline under a single framework.

The three libraries are not mutually exclusive.
A practical setup might use Polars for tabular preprocessing and Daft for multimodal integration.
Choose your tools according to the scale of your project and where your data lives.
