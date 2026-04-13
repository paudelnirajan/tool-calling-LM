# Tool-Calling LM from First Principles

A small decoder-only Transformer trained to map natural-language queries to structured tool calls, built entirely in Python/NumPy.

> **Course:** Intro to NLP · Spring 2026

## Quick Start

### 1. Clone with data

```bash
git clone --recurse-submodules https://github.com/paudelnirajan/tool-calling-LM.git
cd tool-calling-LM
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies from the lock file

```

### 3. Train the model

```bash
# Small config (~1.2M params) — good for a first run
uv run python training/train.py --config small

# Medium config (~2.5M params)
uv run python training/train.py --config medium

# Large config (~4.7M params)
uv run python training/train.py --config large
```

### 4. Train on Google Colab

```python
!pip install uv
!git clone --recurse-submodules https://github.com/paudelnirajan/tool-calling-LM.git
%cd tool-calling-LM
!uv sync
!uv run python training/train.py --config small
```

## Dataset

The dataset is maintained by [Gunabhiram Aruru](https://github.com/AruruGunabhiram) and included as a Git submodule pointing to [`AruruGunabhiram/NLP_Project`](https://github.com/AruruGunabhiram/NLP_Project).

- **10,000 examples** — balanced across 7 tools + NO_CALL
- **Format:** `{input_text, target_text}` sequences ready for training
- **Split:** 8,000 train / 1,000 val / 1,000 test
