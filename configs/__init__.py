"""Load and validate config from JSON files."""

import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent

def load_config(name_or_path):
    """
    Load a config by name ('small', 'medium', 'large') or by file path.

    Returns:
        dict with 'model', 'training', and 'data' sections.
    """
    path = Path(name_or_path)

    # If just a name like "small", look in configs/
    if not path.suffix:
        path = CONFIGS_DIR / f"{name_or_path}.json"

    with open(path) as f:
        config = json.load(f)

    return config


def print_config(config):
    """Pretty-print the config."""
    for section, values in config.items():
        print(f"\n[{section}]")
        for k, v in values.items():
            print(f"  {k:20s} = {v}")
    print()


if __name__ == "__main__":
    for name in ["small", "medium", "large"]:
        print(f"=== {name.upper()} ===")
        cfg = load_config(name)
        print_config(cfg)
