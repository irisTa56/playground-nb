---
paths:
  - "notebooks/daft/**"
---

# Daft Notebook Rules

## `.show()` must be in a separate cell from `print()`

Daft's `.show()` uses an ipywidgets-based renderer that overwrites all prior output in the same Jupyter cell.
Always put `.show()` in its own `# %%` cell, with any preceding `print()` calls in the cell above.

Bad:

```python
# %%
print(f"Found {len(df)} rows")
df.show(5)
```

Good:

```python
# %%
print(f"Found {len(df)} rows")

# %%
df.show(5)
```

## Progress bar suppression

Daft's progress logs are NOT controlled by Python's `logging` module.
To suppress them, use the environment variable:

```python
os.environ["DAFT_PROGRESS_BAR"] = "0"
```

Set this before any `daft.read_*()` or `.collect()` call.
