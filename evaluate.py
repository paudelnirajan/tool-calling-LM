"""
Full evaluation script for the tool-calling Transformer.

Computes six metrics on the test set:
  1. Exact Match (EM)
  2. Tool Selection Accuracy (TSA)
  3. Argument Exact Match (AEM)
  4. Argument F1
  5. Perplexity
  6. NO_CALL Accuracy

Usage:
    python evaluate.py --config small --checkpoint checkpoints/best_model.npz
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import load_config
from model.transformer import ToolCallingLM
from training.tokenizer import Tokenizer, _tokenize_text
from training.train import load_sequences, batches

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
    print(f"Loaded model: {n_params:,} parameters, vocab {tokenizer.vocab_size}")
    return model, tokenizer, cfg


# ── Tool-call parser ──────────────────────────────────────────────────────────

def parse_tool_call(text):
    """
    Parse a tool-call string into (tool_name, args_string).

    Examples:
        'weather(city="Denver", date="tomorrow")'  → ('weather', 'city="Denver", date="tomorrow"')
        'NO_CALL'                                    → ('no_call', '')
        'set_timer(minutes=5)'                       → ('set_timer', 'minutes=5')
    """
    text = text.strip().lower()

    if text.startswith("no_call"):
        return ("no_call", "")

    if "(" in text:
        idx = text.index("(")
        tool = text[:idx].strip()
        args = text[idx + 1 :].rstrip(")").strip()
        return (tool, args)

    return (text, "")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_token_f1(pred_text, gold_text):
    """Compute token-level F1 between two strings."""
    pred_tokens = _tokenize_text(pred_text.lower())
    gold_tokens = _tokenize_text(gold_text.lower())

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1
    gold_set = {}
    for t in gold_tokens:
        gold_set[t] = gold_set.get(t, 0) + 1

    # Count overlapping tokens (bag-of-words intersection)
    overlap = 0
    for t, count in gold_set.items():
        overlap += min(count, pred_set.get(t, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_predictions(model, tokenizer, test_data, max_new_tokens=50):
    """Run greedy generation on all test examples. Returns list of predicted strings."""
    predictions = []
    t0 = time.time()

    for i, ex in enumerate(test_data):
        prompt_ids = tokenizer.encode(ex["input_text"])
        generated = model.generate(
            np.array(prompt_ids, dtype=np.int64),
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
        )
        pred_text = tokenizer.decode(generated[len(prompt_ids) :])
        predictions.append(pred_text)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Generated {i + 1}/{len(test_data)}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Generation complete: {len(test_data)} examples in {elapsed:.1f}s")
    return predictions


def compute_perplexity(model, tokenizer, test_data, cfg):
    """Compute perplexity on the test set using masked loss."""
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    total_loss = 0.0
    total_tokens = 0

    for inp, tgt, mask in batches(
        test_data, tcfg["batch_size"], tokenizer, mcfg["max_seq_len"], shuffle=False
    ):
        logits = model(inp)
        B, T, V = logits.data.shape

        # Softmax (no autograd needed)
        shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=-1, keepdims=True)

        # Per-token NLL
        flat_p = probs.reshape(-1, V)
        flat_t = tgt.reshape(-1)
        flat_m = mask.reshape(-1)
        nll = -np.log(flat_p[np.arange(flat_t.shape[0]), flat_t] + 1e-9)

        total_loss += (nll * flat_m).sum()
        total_tokens += flat_m.sum()

    avg_nll = total_loss / max(total_tokens, 1)
    return float(np.exp(avg_nll))


def evaluate(model, tokenizer, test_data, cfg, max_new_tokens=50):
    """Run all six evaluation metrics and return a results dict."""
    results = {}

    # ── 1. Generate predictions ───────────────────────────────────────────
    print("\nGenerating predictions...")
    predictions = generate_predictions(model, tokenizer, test_data, max_new_tokens)
    golds = [ex["target_text"] for ex in test_data]

    # ── 2. Parse predictions and golds ────────────────────────────────────
    parsed_pred = [parse_tool_call(p) for p in predictions]
    parsed_gold = [parse_tool_call(g) for g in golds]

    # ── 3. Exact Match (EM) ───────────────────────────────────────────────
    em_correct = sum(
        1
        for p, g in zip(predictions, golds)
        if p.strip().lower() == g.strip().lower()
    )
    results["Exact Match (EM)"] = em_correct / len(golds)

    # ── 4. Tool Selection Accuracy (TSA) ──────────────────────────────────
    tsa_correct = sum(
        1 for (pt, _), (gt, _) in zip(parsed_pred, parsed_gold) if pt == gt
    )
    results["Tool Selection Accuracy (TSA)"] = tsa_correct / len(golds)

    # ── 5. Argument Exact Match (AEM) ─────────────────────────────────────
    #    Among examples where the tool is correct AND gold is not NO_CALL
    aem_eligible = 0
    aem_correct = 0
    for (pt, pa), (gt, ga) in zip(parsed_pred, parsed_gold):
        if gt == "no_call":
            continue
        if pt == gt:
            aem_eligible += 1
            if pa.strip() == ga.strip():
                aem_correct += 1
    results["Argument Exact Match (AEM)"] = (
        aem_correct / aem_eligible if aem_eligible > 0 else 0.0
    )

    # ── 6. Argument F1 ───────────────────────────────────────────────────
    f1_scores = []
    for (pt, pa), (gt, ga) in zip(parsed_pred, parsed_gold):
        if gt == "no_call":
            continue
        if pt != gt:
            f1_scores.append(0.0)
        else:
            f1_scores.append(compute_token_f1(pa, ga))
    results["Argument F1"] = (
        sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    )

    # ── 7. Perplexity ────────────────────────────────────────────────────
    print("Computing perplexity...")
    results["Perplexity"] = compute_perplexity(model, tokenizer, test_data, cfg)

    # ── 8. NO_CALL Accuracy ──────────────────────────────────────────────
    nc_total = 0
    nc_correct = 0
    for (pt, _), (gt, _) in zip(parsed_pred, parsed_gold):
        if gt == "no_call":
            nc_total += 1
            if pt == "no_call":
                nc_correct += 1
    results["NO_CALL Accuracy"] = nc_correct / nc_total if nc_total > 0 else 0.0

    return results, predictions


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_results(results):
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for name, value in results.items():
        if name == "Perplexity":
            print(f"  {name:<35s} {value:>10.2f}")
        else:
            print(f"  {name:<35s} {value:>10.1%}")
    print("=" * 50)


def print_sample_predictions(test_data, predictions, n=10):
    """Print a few sample predictions for qualitative inspection."""
    print(f"\n{'─' * 60}")
    print(f"  SAMPLE PREDICTIONS (first {n})")
    print(f"{'─' * 60}")
    for i in range(min(n, len(test_data))):
        # Extract just the user query from input_text
        inp = test_data[i]["input_text"]
        query = inp.split("\n")[1] if "\n" in inp else inp  # "User: ..."
        gold = test_data[i]["target_text"]
        pred = predictions[i]
        match = "  " if pred.strip().lower() == gold.strip().lower() else "X "
        print(f"\n  [{match}] {query}")
        print(f"       Gold: {gold}")
        print(f"       Pred: {pred}")
    print(f"\n{'─' * 60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the tool-calling model")
    parser.add_argument(
        "--config", default="small", help="Config name or path (default: small)"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.npz",
        help="Path to model checkpoint (default: checkpoints/best_model.npz)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max tokens to generate per example (default: 50)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of sample predictions to print (default: 10)",
    )
    args = parser.parse_args()

    # Load model
    ckpt_path = ROOT / args.checkpoint
    model, tokenizer, cfg = load_model_and_tokenizer(args.config, ckpt_path)

    # Load test data
    dcfg = cfg["data"]
    split_file = dcfg["val_file"] if args.split == "val" else dcfg["test_file"]
    test_data = load_sequences(ROOT / split_file)
    print(f"Evaluating on {args.split} split: {len(test_data)} examples")

    # Run evaluation
    results, predictions = evaluate(
        model, tokenizer, test_data, cfg, max_new_tokens=args.max_new_tokens
    )

    # Print results
    print_results(results)
    print_sample_predictions(test_data, predictions, n=args.samples)


if __name__ == "__main__":
    main()
