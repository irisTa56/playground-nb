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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/daft/image_color_query.ipynb)
#
# # Image Color Query — Top-N Reddest Images
#
# Find the most red-colored images from the
# [Open Images](https://storage.googleapis.com/openimages/web/index.html) dataset
# using **Daft** with anonymous S3 access.
#
# Based on the [Daft tutorial: Top N Most Red Images](https://github.com/Eventual-Inc/Daft/blob/main/tutorials/image_querying/top_n_red_color.ipynb),
# adapted for Daft 0.7+ APIs.
#
# This notebook is **self-contained**: it installs its own dependencies and
# downloads data from S3 — no prior setup or credentials are needed.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "daft[aws]>=0.7",
#   "numpy>=2.4",
#   "Pillow>=12.1",
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
for pkg in ("daft", "numpy", "Pillow"):
    print(f"  {pkg}: {importlib.metadata.version(pkg)}")

# %% [markdown]
# ## 1. Load Image Paths from S3
#
# Daft's `from_glob_path` lists objects in S3 using anonymous access. We use the
# `daft-public-data` mirror of Open Images validation images — no AWS credentials
# or region override needed.

# %%
import daft
from daft import DataType
from daft.io import IOConfig, S3Config

TOP_N = 10
MAX_IMAGES = 100  # limit for local execution; increase for larger runs

IO_CONFIG = IOConfig(s3=S3Config(anonymous=True))

df = daft.from_glob_path(
    "s3://daft-public-data/open-images/validation-images/*",
    io_config=IO_CONFIG,
)

# Keep only reasonably sized images (200–300 KB) to speed up processing
df = df.where(df["size"].between(200_000, 300_000)).limit(MAX_IMAGES)

print(f"Selected up to {MAX_IMAGES} images (200–300 KB each)")

# %%
df.show(5)

# %% [markdown]
# ## 2. Download and Decode Images
#
# Daft can download file contents directly from S3 paths via `.download()`,
# then decode bytes into image objects with `.decode_image()`.

# %%
df = df.with_column("image", df["path"].download(io_config=IO_CONFIG))
df = df.with_column("image", df["image"].decode_image())

df = df.collect()
print(f"Downloaded and decoded {len(df)} images")

# %%
df.show(3)

# %% [markdown]
# ## 3. Detect Red Pixels
#
# We convert each image to HSV color space and identify pixels where the hue falls
# in the red range (accounting for hue wrap-around) with sufficient saturation and
# value. The "redness score" is the fraction of red pixels in the image.

# %%
import numpy as np
from PIL import Image


def compute_redness(image_arr: np.ndarray) -> float:
    """Compute fraction of red pixels in an image using HSV thresholds.

    Daft passes decoded images as RGB numpy arrays to .apply().
    """
    hsv = np.array(Image.fromarray(image_arr).convert("HSV"))
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # PIL HSV maps hue 0–360° to 0–255
    # Low red: hue < 10 (≈0–14°), High red: hue > 240 (≈338–360°)
    red_mask = ((h < 10) | (h > 240)) & (s > 50) & (v > 50)
    total = red_mask.size
    if total == 0:
        return 0.0
    return float(red_mask.sum()) / total


df = df.with_column(
    "redness",
    df["image"].apply(compute_redness, return_dtype=DataType.float64()),
)

df = df.collect()
print("Redness scores computed")

# %% [markdown]
# ## 4. Top-N Reddest Images
#
# Sort by redness score in descending order and display the top results. Daft
# renders Image columns inline in Jupyter automatically.

# %%
top_red = df.sort("redness", desc=True).limit(TOP_N)
top_red = top_red.select("path", "redness", "image")
top_red = top_red.collect()

print(f"Top {TOP_N} reddest images:")

# %%
top_red.show(TOP_N)

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# - **Anonymous S3 access** with Daft's `IOConfig(s3=S3Config(anonymous=True))`
# - **Image downloading and decoding** using Daft's built-in URL and image accessors
# - **Custom column-wise computation** via `.apply()` with a Python function
# - **Lazy evaluation** — filters, downloads, and computations are optimised before
#   execution
# - **Built-in image rendering** — Daft displays Image columns natively in Jupyter
