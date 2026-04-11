"""Utility to load data from the submodule."""

import json
from pathlib import Path

# Resolve data directory relative to this file's location
DATA_DIR = Path(__file__).resolve().parent.parent / "data_repo" / "data"


def load_split(split: str) -> list[dict]:
    """Load a dataset split (train, val, or test) in sequence format.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        List of dicts with 'input_text' and 'target_text' keys.
    """
    path = DATA_DIR / f"{split}_sequences.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Did you run `git submodule update --init`?"
        )
    with open(path) as f:
        return json.load(f)


def load_tools() -> list[dict]:
    """Load the canonical tool schema."""
    with open(DATA_DIR / "tools.json") as f:
        return json.load(f)


def load_raw_split(split: str) -> list[dict]:
    """Load a raw dataset split (before sequence formatting).

    Args:
        split: One of 'train', 'val', 'test', or 'dataset_full'.

    Returns:
        List of dicts with 'query', 'tool', and 'arguments' keys.
    """
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Did you run `git submodule update --init`?"
        )
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Quick sanity check
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    tools = load_tools()

    print(f"Tools:  {len(tools)} defined")
    print(f"Train:  {len(train)} examples")
    print(f"Val:    {len(val)} examples")
    print(f"Test:   {len(test)} examples")
    print(f"\nSample input:\n{train[0]['input_text'][:200]}...")
    print(f"\nSample target:\n{train[0]['target_text']}")
