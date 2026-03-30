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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/lance/clip_multimodal_lance.ipynb)
#
# # CLIP Multimodal ML with Lance
#
# A complete multimodal ML workflow using [Lance](https://lancedb.github.io/lance/)
# as the columnar data format:
#
# 1. **Dataset creation** — download Flickr8k, parse annotations, store images +
#    captions in a Lance dataset.
# 2. **CLIP training** — build image/text encoders, train with contrastive loss,
#    reading data from Lance.
# 3. **Artifact management** — save/load/version PyTorch model checkpoints as
#    Lance datasets.
#
# Based on the official Lance examples:
#
# - [Flickr8k Dataset Creation](https://lance.org/examples/python/flickr8k_dataset_creation/)
# - [CLIP Training](https://lance.org/examples/python/clip_training/)
# - [Artifact Management](https://lance.org/examples/python/artifact_management/)
#
# This notebook is **self-contained**: it installs its own dependencies, downloads
# the dataset, and detects the available accelerator — no prior setup is needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pylance>=3.0",
#   "pyarrow>=23.0",
#   "torch>=2.11",
#   "torchvision>=0.26",
#   "timm>=1.0",
#   "transformers>=5.4",
#   "tqdm>=4.67",
#   "matplotlib>=3.10",
#   "numpy>=2.4",
#   "Pillow>=12.1",
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
for pkg in ("pylance", "pyarrow", "torch", "timm", "transformers", "pillow"):
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
# ## 1. Flickr8k Dataset Creation (Lance)
#
# The [Flickr8k](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
# dataset contains ~8,000 images with 5 captions each.
# We download it, parse the annotations, and store everything in a
# [Lance](https://lancedb.github.io/lance/) columnar dataset.
#
# Lance provides efficient random access to binary blobs (images) alongside
# structured columns (captions), with built-in versioning for free.

# %%
import shutil
import urllib.request
import zipfile

DATA_DIR = Path("data/flickr8k")
LANCE_DIR = Path("data/flickr8k_lance")

FLICKR_DATASET_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
FLICKR_TEXT_URL = (
    "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
)


def _download_and_extract(url: str, dest: Path) -> None:
    """Download a zip file and extract it to *dest*."""
    zip_path = dest / Path(url).name
    if not zip_path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()


if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)
    _download_and_extract(FLICKR_DATASET_URL, DATA_DIR)
    _download_and_extract(FLICKR_TEXT_URL, DATA_DIR)

print(f"Data directory: {DATA_DIR}")

# %% [markdown]
# ### Parse annotations
#
# The caption file (`Flickr8k.token.txt`) has one caption per line in the format
# `image_id#index\tcaption`.
# We group the 5 captions per image into a dict.

# %%
from collections import defaultdict

caption_file = DATA_DIR / "Flickr8k_text" / "Flickr8k.token.txt"
# Flickr8k may also extract with this layout:
if not caption_file.exists():
    caption_file = DATA_DIR / "Flickr8k.token.txt"

captions_by_image: dict[str, list[str]] = defaultdict(list)

for line in caption_file.read_text().strip().splitlines():
    # Format: image_id#index\tcaption
    image_cap_id, caption = line.split("\t", maxsplit=1)
    image_id = image_cap_id.rsplit("#", maxsplit=1)[0]
    captions_by_image[image_id].append(caption.strip())

print(f"Parsed {len(captions_by_image)} images with captions")
print("Example captions for first image:")
first_id = next(iter(captions_by_image))
for i, cap in enumerate(captions_by_image[first_id]):
    print(f"  [{i}] {cap}")

# %% [markdown]
# ### Write the Lance dataset
#
# Lance adopts the **Apache Arrow** memory layout internally, so `pa.schema()`,
# `pa.RecordBatch`, and `pa.RecordBatchReader` work directly with
# `lance.write_dataset()` — no conversion layer is needed.
# This also means Lance datasets interoperate natively with the wider Arrow
# ecosystem (Pandas, Polars, DuckDB, etc.).
#
# We use a generator of `pa.RecordBatch` objects piped through
# `pa.RecordBatchReader` to `lance.write_dataset()`.
# This keeps memory usage low — we never hold all images in RAM at once.

# %%
from collections.abc import Iterator

import lance
import pyarrow as pa

SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string()),
        pa.field("image", pa.binary()),
        pa.field("captions", pa.list_(pa.string())),
    ]
)

IMAGE_DIR = DATA_DIR / "Flicker8k_Dataset"
if not IMAGE_DIR.exists():
    IMAGE_DIR = DATA_DIR / "Flickr8k_Dataset"


def _flickr8k_batches(batch_size: int = 256) -> Iterator[pa.RecordBatch]:
    """Yield RecordBatch objects from Flickr8k images + captions."""
    ids: list[str] = []
    images: list[bytes] = []
    caps: list[list[str]] = []

    for image_id, cap_list in captions_by_image.items():
        image_path = IMAGE_DIR / image_id
        if not image_path.exists():
            continue
        ids.append(image_id)
        images.append(image_path.read_bytes())
        caps.append(cap_list)

        if len(ids) >= batch_size:
            yield pa.RecordBatch.from_pydict(
                {"image_id": ids, "image": images, "captions": caps}, schema=SCHEMA
            )
            ids, images, caps = [], [], []

    if ids:
        yield pa.RecordBatch.from_pydict(
            {"image_id": ids, "image": images, "captions": caps}, schema=SCHEMA
        )


reader = pa.RecordBatchReader.from_batches(SCHEMA, _flickr8k_batches())
lance.write_dataset(reader, LANCE_DIR, schema=SCHEMA)
print(f"Wrote Lance dataset to {LANCE_DIR}")

# %% [markdown]
# ### Verify the dataset

# %%
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ds = lance.dataset(LANCE_DIR)
print(f"Row count: {ds.count_rows()}")
print(f"Schema:\n{ds.schema}")

sample = ds.take([0]).to_pydict()
img = Image.open(io.BytesIO(sample["image"][0]))
print(f"\nSample image: {sample['image_id'][0]}  ({img.size[0]}x{img.size[1]})")
print("Captions:")
for cap in sample["captions"][0]:
    print(f"  • {cap}")

plt.figure(figsize=(4, 4))
plt.imshow(np.array(img))
plt.axis("off")
plt.title(sample["image_id"][0])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. CLIP Model Definition & Training
#
# [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language–Image Pre-training)
# learns a shared embedding space for images and text.
# An image encoder and a text encoder are trained with **InfoNCE contrastive loss**:
# matching image-text pairs are pulled together while non-matching pairs are pushed
# apart.
#
# We use **ResNet-50** (via `timm`) as the image encoder and **BERT-base-cased**
# (via `transformers`) as the text encoder, each followed by a projection head that
# maps to a shared embedding dimension.

# %%
import math
from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    embed_dim: int = 256
    epochs: int = 3
    lr_head: float = 1e-3
    lr_image: float = 1e-4
    lr_text: float = 1e-5


CFG = Config()

# %% [markdown]
# ### Dataset — reading from Lance

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


class CLIPLanceDataset(Dataset):
    """Read images and captions from a Lance dataset for CLIP training."""

    def __init__(self, lance_path: str | Path, transform: transforms.Compose) -> None:
        self._ds = lance.dataset(lance_path)
        self._len = self._ds.count_rows()
        self._transform = transform

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx):  # type: ignore[override]
        row = self._ds.take([idx]).to_pydict()
        img = Image.open(io.BytesIO(row["image"][0])).convert("RGB")
        img_tensor = self._transform(img)

        # Use the longest caption for training
        caption = max(row["captions"][0], key=len)
        tokens = tokenizer(  # type: ignore[misc]
            caption,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        return {
            "image": img_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


train_transform = transforms.Compose(
    [
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.CenterCrop(CFG.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CLIPLanceDataset(LANCE_DIR, train_transform)
print(f"Dataset size: {len(train_dataset)}")

# %% [markdown]
# ### Model architecture

# %%
import timm


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 2048)


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained("bert-base-cased")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS token (B, 768)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.linear2(self.gelu(self.linear1(x)))
        return self.norm(projected)


class CLIPModel(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_proj = ProjectionHead(2048, embed_dim)
        self.text_proj = ProjectionHead(768, embed_dim)
        # Learnable temperature (log-parameterized, per CLIP paper)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_features = self.image_encoder(images)
        txt_features = self.text_encoder(input_ids, attention_mask)

        img_embeds = nn.functional.normalize(self.image_proj(img_features), dim=-1)
        txt_embeds = nn.functional.normalize(self.text_proj(txt_features), dim=-1)

        # Clamp temperature to avoid instability
        temperature = self.log_temperature.exp().clamp(max=100.0)
        return img_embeds, txt_embeds, temperature


# %% [markdown]
# ### Contrastive loss (InfoNCE)

# %%
import torch.nn.functional as F


def clip_loss(
    img_embeds: torch.Tensor, txt_embeds: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    """Symmetric CLIP contrastive loss (InfoNCE)."""
    logits = img_embeds @ txt_embeds.T * temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


# %% [markdown]
# ### Save & load helpers
#
# We define these before the training loop so checkpoints can be saved after each
# epoch.
# Each parameter tensor is flattened to a 1-D float64 array and stored alongside
# its shape in a Lance dataset.
# Successive `mode="overwrite"` writes create new Lance versions while keeping
# old versions accessible via `lance.dataset(path, version=N)`.

# %%
from collections import OrderedDict

MODEL_SCHEMA = pa.schema(
    [
        pa.field("name", pa.string()),
        pa.field("value", pa.list_(pa.float64())),
        pa.field("shape", pa.list_(pa.int64())),
    ]
)


def save_model(state_dict: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save a PyTorch state_dict as a Lance dataset (one row per parameter)."""

    def _batches() -> Iterator[pa.RecordBatch]:
        for name, tensor in state_dict.items():
            flat = tensor.detach().cpu().to(torch.float64).flatten().tolist()
            shape = list(tensor.shape)
            yield pa.RecordBatch.from_pydict(
                {"name": [name], "value": [flat], "shape": [shape]},
                schema=MODEL_SCHEMA,
            )

    reader = pa.RecordBatchReader.from_batches(MODEL_SCHEMA, _batches())
    mode = "overwrite" if Path(path).exists() else "create"
    lance.write_dataset(reader, path, schema=MODEL_SCHEMA, mode=mode)


def load_model(
    path: str | Path, version: int | None = None
) -> OrderedDict[str, torch.Tensor]:
    """Load a state_dict from a versioned Lance dataset."""
    ds = lance.dataset(path, version=version)
    table = ds.to_table()
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for i in range(table.num_rows):
        name = table.column("name")[i].as_py()
        flat = table.column("value")[i].as_py()
        shape = table.column("shape")[i].as_py()
        tensor = torch.tensor(flat, dtype=torch.float32).reshape(shape)
        state_dict[name] = tensor
    return state_dict


# %% [markdown]
# ### Training loop

# %%
ARTIFACT_DIR = Path("data/clip_model_lance")

from tqdm.auto import tqdm  # noqa: E402


def train(model: CLIPModel, dataset: CLIPLanceDataset, cfg: Config) -> CLIPModel:
    """Train CLIP model and save checkpoints as Lance datasets after each epoch."""
    # num_workers=0 avoids fork-related segfaults when Lance datasets are
    # accessed from worker subprocesses (both macOS and Colab/Linux).
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_proj.parameters(), "lr": cfg.lr_head},
            {"params": model.text_proj.parameters(), "lr": cfg.lr_head},
            {"params": model.log_temperature, "lr": cfg.lr_head},
            {"params": model.image_encoder.parameters(), "lr": cfg.lr_image},
            {"params": model.text_encoder.parameters(), "lr": cfg.lr_text},
        ]
    )

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            img_embeds, txt_embeds, temperature = model(
                images, input_ids, attention_mask
            )
            loss = clip_loss(img_embeds, txt_embeds, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch (creates a new Lance version)
        save_model(model.state_dict(), ARTIFACT_DIR)
        print(
            f"  Saved checkpoint (Lance version {lance.dataset(ARTIFACT_DIR).version})"
        )

    return model


model = CLIPModel(embed_dim=CFG.embed_dim).to(DEVICE)
model = train(model, train_dataset, CFG)
print("Training complete.")

# %% [markdown]
# ## 3. Artifact Management — Model Versioning with Lance
#
# The training loop above saved a checkpoint after each epoch.
# Each `save_model` call wrote a new Lance version, so we can now load any
# epoch's weights by version number and compare them.

# %% [markdown]
# ### Demonstrate versioning

# %%
artifact_ds = lance.dataset(ARTIFACT_DIR)
latest_version = artifact_ds.version
print(f"Lance artifact dataset: {ARTIFACT_DIR}")
print(f"  Versions available: 1 .. {latest_version}")

# Load version 1 (epoch 1) and the latest version (final epoch)
sd_v1 = load_model(ARTIFACT_DIR, version=1)
sd_latest = load_model(ARTIFACT_DIR, version=latest_version)

# Compare a parameter to show they differ
param_name = next(iter(sd_v1))
diff = (sd_v1[param_name] - sd_latest[param_name]).abs().mean().item()
print(f"\n  Parameter '{param_name}':")
print(f"    Mean absolute difference (v1 vs v{latest_version}): {diff:.6f}")

# Verify round-trip: load version 1 into a fresh model
fresh_model = CLIPModel(embed_dim=CFG.embed_dim)
fresh_model.load_state_dict(sd_v1)
print("\n  Successfully loaded version 1 into a fresh CLIPModel")

# %% [markdown]
# ## Cleanup

# %%
for d in (DATA_DIR, LANCE_DIR, ARTIFACT_DIR):
    if d.exists():
        shutil.rmtree(d)
        print(f"Removed {d}")
