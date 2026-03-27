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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irisTa56/playground-nb/blob/main/notebooks/daft/text_to_image_generation.ipynb)
#
# # Text-to-Image Generation with Stable Diffusion
#
# Generate images from text prompts using the **Stable Diffusion** model,
# orchestrated by **Daft** for data loading and pipeline execution.
#
# Based on the [Daft tutorial: Text-to-Image Generation](https://github.com/Eventual-Inc/Daft/blob/main/tutorials/text_to_image/text_to_image_generation.ipynb),
# adapted for Daft 0.7+ APIs.
#
# This notebook is **self-contained**: it installs its own dependencies and
# reads data from S3 — no prior setup or credentials are needed.
#
# > **Note:** The Stable Diffusion model download is **~4 GB**. First run will
# > take extra time while the model weights are fetched from HuggingFace Hub.

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "daft[aws]>=0.7",
#   "transformers>=4.51",
#   "diffusers>=0.33",
#   "torch>=2.11",
#   "torchvision>=0.26",
#   "accelerate>=1.13",
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
for pkg in ("daft", "diffusers", "torch"):
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
if DEVICE.type == "cpu":
    print(
        "WARNING: Stable Diffusion on CPU is extremely slow (~5-10 min per image). "
        "A CUDA GPU is strongly recommended."
    )

# %% [markdown]
# ## 1. Load the Dataset
#
# We load a LAION aesthetic-score parquet file from S3. Each row contains an
# image `URL`, a `TEXT` description, and an `AESTHETIC_SCORE`.

# %%
import os

os.environ["DAFT_PROGRESS_BAR"] = "0"

import daft
from daft.io import IOConfig, S3Config

NUM_IMAGES = 3
MODEL_ID = "runwayml/stable-diffusion-v1-5"

PARQUET_PATH = (
    "s3://daft-public-data/tutorials/laion-parquet/"
    "train-00000-of-00001-6f24a7497df494ae.parquet"
)
IO_CONFIG = IOConfig(s3=S3Config(anonymous=True, region_name="us-west-2"))

parquet_df = daft.read_parquet(PARQUET_PATH, io_config=IO_CONFIG)
parquet_df = parquet_df.collect()

print(f"Loaded {len(parquet_df)} rows from LAION parquet")

# %%
parquet_df.show(5)

# %% [markdown]
# ## 2. Preview Source Images
#
# Filter for longer descriptions (>50 chars), download a few source images, and
# display them to get a sense of the dataset.

# %%
parquet_df = parquet_df.select("URL", "TEXT", "AESTHETIC_SCORE")

images_df = parquet_df.where(parquet_df["TEXT"].length() > 50)  # type: ignore[operator]
images_df = images_df.with_column(
    "image",
    images_df["URL"].download(on_error="null").decode_image(on_error="null"),
)
images_df = images_df.limit(5).where(daft.col("image").not_null())
images_df = images_df.collect()

print(f"Downloaded {len(images_df)} sample images")

# %%
images_df.show(5)

# %% [markdown]
# ## 3. Generate Images with Stable Diffusion
#
# We initialize the `StableDiffusionPipeline` once and use it inside a
# `@daft.func` UDF to generate an image for each text prompt.
#
# The pipeline uses `float16` on CUDA for speed, and `float32` otherwise.

# %%
from diffusers import StableDiffusionPipeline

dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing(1)

print(f"Loaded {MODEL_ID} on {DEVICE} (dtype={dtype})")


@daft.func(return_dtype=daft.DataType.python())
def generate_image(text: str):
    return pipe(text, num_inference_steps=20, height=512, width=512).images[0]


# Select top-scoring prompts with longer descriptions
prompts_df = parquet_df.where(parquet_df["TEXT"].length() > 50)  # type: ignore[operator]
prompts_df = prompts_df.distinct("TEXT")
prompts_df = prompts_df.sort("AESTHETIC_SCORE", desc=True).limit(NUM_IMAGES)
prompts_df = prompts_df.with_column(
    "generated_image", generate_image(prompts_df["TEXT"])
)
prompts_df = prompts_df.collect()

print(f"Generated {len(prompts_df)} images")

# %% [markdown]
# ## 4. Display Generated Images
#
# Since the generated images are Python objects (PIL Images) stored in a
# `DataType.python()` column, we extract them and display with
# `IPython.display`.

# %%
from IPython.display import display

rows = prompts_df.to_pydict()
for i, (text, img) in enumerate(zip(rows["TEXT"], rows["generated_image"])):
    print(f"\n--- Image {i + 1} ---")
    print(f"Prompt: {text}")
    display(img)

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# - **Loading parquet from S3** with anonymous access via `daft.read_parquet()`
# - **Image downloading and decoding** using Daft's built-in `.download()` and
#   `.decode_image()` methods
# - **Stable Diffusion inference** with `StableDiffusionPipeline` from
#   HuggingFace `diffusers`, running on the best available device
# - **UDFs in Daft** — `@daft.func` wrapping the pipeline for row-wise
#   image generation
# - **Displaying generated PIL images** via `IPython.display.display()`
