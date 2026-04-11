"""
Main training script for the tool-calling Transformer.

Usage:
    python training/train.py --config small
    python training/train.py --config configs/medium.json
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

# project imports
from configs import load_config, print_config
from model.transformer import ToolCallingLM, cross_entropy_loss
from training.tokenizer import Tokenizer
from training.optimizer import Adam, CosineScheduler

ROOT = Path(__file__).resolve().parent.parent


def load_sequences(path):
    with open(path) as f:
        return json.load(f)


def prepare_batch(examples, tokenizer, max_len):
    """
    Tokenize a list of examples and return padded input/target arrays.

    For each example the full sequence is:
        input_text + " " + target_text + <EOS>

    We train autoregressively: input = seq[:-1], target = seq[1:]
    """
    input_ids, target_ids = [], []

    for ex in examples:
        full = ex["input_text"] + " " + ex["target_text"]
        ids = tokenizer.encode(full, add_eos=True)
        if len(ids) > max_len:
            ids = ids[:max_len]
        input_ids.append(ids[:-1])
        target_ids.append(ids[1:])

    # Pad to the longest sequence in this batch
    max_t = max(len(s) for s in input_ids)
    pad = tokenizer.pad_id

    inp = np.full((len(examples), max_t), pad, dtype=np.int64)
    tgt = np.full((len(examples), max_t), pad, dtype=np.int64)

    for i, (x, y) in enumerate(zip(input_ids, target_ids)):
        inp[i, : len(x)] = x
        tgt[i, : len(y)] = y

    return inp, tgt


def batches(data, batch_size, tokenizer, max_len, shuffle=True):
    """Yield (input, target) batches from the dataset."""
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(data), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = [data[i] for i in batch_idx]
        yield prepare_batch(batch, tokenizer, max_len)


def quick_eval(model, tokenizer, val_data, n=50):
    """Check tool selection accuracy on a small sample."""
    correct = 0
    sample = val_data[:n]

    for ex in sample:
        prompt_ids = tokenizer.encode(ex["input_text"])
        generated = model.generate(
            np.array(prompt_ids, dtype=np.int64),
            max_new_tokens=30,
            eos_id=tokenizer.eos_id,
        )
        pred = tokenizer.decode(generated[len(prompt_ids):])
        gold = ex["target_text"]

        # Extract tool name (everything before the first '(' or just NO_CALL)
        pred_tool = pred.split("(")[0].strip()
        gold_tool = gold.split("(")[0].strip()

        if pred_tool == gold_tool:
            correct += 1

    return correct / len(sample)


def train(config_name="small"):
    cfg = load_config(config_name)
    print_config(cfg)

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    print("Loading data...")
    train_data = load_sequences(ROOT / dcfg["train_file"])
    val_data = load_sequences(ROOT / dcfg["val_file"])
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}")
    print("Building tokenizer...")
    tokenizer = Tokenizer.from_data_files(
        ROOT / dcfg["train_file"],
        ROOT / dcfg["val_file"],
    )
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Update vocab_size in config to match actual tokenizer
    mcfg["vocab_size"] = tokenizer.vocab_size

    model = ToolCallingLM.from_config(mcfg)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    optimizer = Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=(tcfg["beta1"], tcfg["beta2"]),
        eps=tcfg["eps"],
        grad_clip=tcfg["grad_clip"],
    )

    steps_per_epoch = len(train_data) // tcfg["batch_size"]
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = CosineScheduler(optimizer, tcfg["warmup_steps"], total_steps)

    print(f"\nTraining for {tcfg['epochs']} epochs ({total_steps} steps)...\n")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, tcfg["epochs"] + 1):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for inp, tgt in batches(train_data, tcfg["batch_size"], tokenizer, mcfg["max_seq_len"]):
            # Forward
            logits = model(inp)

            # Mask out padding from loss
            loss = cross_entropy_loss(logits, tgt)

            # Backward
            model.zero_grad()
            loss.backward()

            # Update
            lr = scheduler.step(global_step)
            optimizer.step()

            epoch_loss += float(loss.data)
            n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                print(f"  step {global_step:5d}  loss={float(loss.data):.4f}  lr={lr:.2e}")

        avg_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        val_loss = 0.0
        val_batches = 0
        for inp, tgt in batches(val_data, tcfg["batch_size"], tokenizer, mcfg["max_seq_len"], shuffle=False):
            logits = model(inp)
            loss = cross_entropy_loss(logits, tgt)
            val_loss += float(loss.data)
            val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1)
        tsa = quick_eval(model, tokenizer, val_data, n=50)

        print(
            f"Epoch {epoch:2d}/{tcfg['epochs']}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"tool_acc={tsa:.1%}  "
            f"time={elapsed:.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_dir = ROOT / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            np.savez(
                ckpt_dir / "best_model.npz",
                **{f"p{i}": p.data for i, p in enumerate(model.parameters())},
            )
            tokenizer.save(ckpt_dir / "tokenizer.json")
            print(f"  ✓ Saved best checkpoint (val_loss={avg_val_loss:.4f})")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="small", help="Config name or path")
    args = parser.parse_args()
    train(args.config)
