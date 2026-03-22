# Playground for Notebook

This repository is a playground for testing and experimenting with Jupyter Notebooks.

## Setup

```shell
mise install
uv venv --python "$(mise which python)"
```

## Development

You can use [mise to set up pre-commit hooks](https://mise.jdx.dev/cli/generate/git-pre-commit.html) for this repository.

```shell
mise generate git-pre-commit --write --task=pre-commit
```
