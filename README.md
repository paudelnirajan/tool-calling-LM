# Tool-Calling LM from First Principles

A small decoder-only Transformer trained to map natural-language queries to structured tool calls, built entirely in Python/NumPy.

> **Course:** Intro to NLP · Spring 2026

## Quick Start

### 1. Clone with data

```bash
git clone --recurse-submodules https://github.com/<your-username>/tool-calling-LM.git
cd tool-calling-LM
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Verify the dataset

```bash
python data_repo/data_generation/check_dataset.py
```

## Dataset

The dataset is maintained by [Gunabhiram Aruru](https://github.com/AruruGunabhiram) and included as a Git submodule pointing to [`AruruGunabhiram/NLP_Project`](https://github.com/AruruGunabhiram/NLP_Project).

- **10,000 examples** — balanced across 7 tools + NO_CALL
- **Format:** `{input_text, target_text}` sequences ready for training
- **Split:** 8,000 train / 1,000 val / 1,000 test

