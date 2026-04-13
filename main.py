"""
Interactive demo for the tool-calling Transformer.

Load a trained checkpoint and predict tool calls from natural-language queries.

Usage:
    # Single query
    python main.py --config small --checkpoint checkpoints/best_model.npz \
                   --query "What's the weather in Denver tomorrow?"

    # Interactive mode
    python main.py --config small --checkpoint checkpoints/best_model.npz
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import load_config
from model.transformer import ToolCallingLM
from training.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_model_and_tokenizer(config_name, checkpoint_path):
    """Load a trained model and its tokenizer from a checkpoint."""
    cfg = load_config(config_name)
    mcfg = cfg["model"]

    ckpt_dir = Path(checkpoint_path).parent
    tokenizer = Tokenizer.load(ckpt_dir / "tokenizer.json")
    mcfg["vocab_size"] = tokenizer.vocab_size

    model = ToolCallingLM.from_config(mcfg)
    ckpt = np.load(checkpoint_path)
    params = model.parameters()
    for i, p in enumerate(params):
        p.data = ckpt[f"p{i}"]

    n_params = sum(p.data.size for p in params)
    print(f"Model loaded: {n_params:,} parameters, vocab {tokenizer.vocab_size}")
    return model, tokenizer


# ── Prompt construction ───────────────────────────────────────────────────────

def load_tool_signatures():
    """Load tool schemas and build the 'Available tools:' string."""
    tools_path = ROOT / "data_repo" / "data" / "tools.json"
    with open(tools_path) as f:
        tools = json.load(f)

    sigs = []
    for tool in tools:
        args = ",".join(tool["arguments"].keys())
        sigs.append(f"{tool['name']}({args})")
    return ", ".join(sigs)


def build_prompt(query, tool_string):
    """
    Build a model prompt matching the training data format:
        <BOS>\\nUser: {query}\\nAvailable tools: {tools}\\n<CALL>
    """
    return f"<BOS>\nUser: {query}\nAvailable tools: {tool_string}\n<CALL>"


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, tokenizer, query, tool_string, max_new_tokens=50):
    """Generate a tool-call prediction for a user query."""
    prompt = build_prompt(query, tool_string)
    prompt_ids = tokenizer.encode(prompt)

    generated = model.generate(
        np.array(prompt_ids, dtype=np.int64),
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
    )

    output = tokenizer.decode(generated[len(prompt_ids) :])
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tool-calling LM demo")
    parser.add_argument(
        "--config", default="small", help="Config name or path (default: small)"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.npz",
        help="Path to model checkpoint (default: checkpoints/best_model.npz)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to predict (omit for interactive mode)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max tokens to generate (default: 50)",
    )
    args = parser.parse_args()

    ckpt_path = ROOT / args.checkpoint
    model, tokenizer = load_model_and_tokenizer(args.config, ckpt_path)
    tool_string = load_tool_signatures()

    # ── Single query mode ─────────────────────────────────────────────────
    if args.query:
        result = predict(model, tokenizer, args.query, tool_string, args.max_new_tokens)
        print(f"\nQuery:      {args.query}")
        print(f"Prediction: {result}")
        return

    # ── Interactive mode ──────────────────────────────────────────────────
    print("\nAvailable tools:")
    tools_path = ROOT / "data_repo" / "data" / "tools.json"
    with open(tools_path) as f:
        tools = json.load(f)
    for tool in tools:
        args_str = ", ".join(
            f"{k}: {v}" for k, v in tool["arguments"].items()
        )
        print(f"  {tool['name']}({args_str})")
    print(f"\nType a query and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = predict(model, tokenizer, query, tool_string)
        print(f"LM:  {result}\n")


if __name__ == "__main__":
    main()
