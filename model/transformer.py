"""
Decoder-only Transformer for tool-calling — device-agnostic.

Assembles the layers into a full language model and provides
a cross-entropy loss function and a simple greedy-generate method.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as _np
from backend import xp, to_numpy, from_numpy
from .autograd import Tensor
from .layers import Module, Embedding, LayerNorm, Linear, TransformerBlock


def sinusoidal_encoding(max_len, d_model):
    # Built in plain numpy then moved to device (computed once at startup)
    pe = _np.zeros((max_len, d_model), dtype=_np.float32)
    pos = _np.arange(max_len)[:, None]
    div = _np.exp(_np.arange(0, d_model, 2) * -(_np.log(10000.0) / d_model))
    pe[:, 0::2] = _np.sin(pos * div)
    pe[:, 1::2] = _np.cos(pos * div)
    return from_numpy(pe)   # move to active device


def cross_entropy_loss(logits, targets, mask=None):
    """
    Args:
        logits:  Tensor of shape (B, T, V)  — raw model output
        targets: array of shape (B, T)       — integer token IDs
        mask:    array of shape (B, T)       — 1.0 include, 0.0 ignore
                 (padding + prompt tokens).  None → include all.

    Returns:
        Scalar Tensor (mean loss over unmasked positions).
    """
    B, T, V = logits.shape

    # Always work in numpy for the loss scalar computation
    logits_np = to_numpy(logits.data)
    targets_np = _np.asarray(to_numpy(targets), dtype=_np.int64)

    if mask is None:
        mask_np = _np.ones((B, T), dtype=_np.float32)
    else:
        mask_np = _np.asarray(to_numpy(mask), dtype=_np.float32)

    # Numerically stable softmax
    shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
    probs = _np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)

    # Masked NLL
    flat_p = probs.reshape(-1, V)
    flat_t = targets_np.reshape(-1)
    flat_m = mask_np.reshape(-1)
    N = max(float(flat_m.sum()), 1.0)

    per_token = -_np.log(flat_p[_np.arange(flat_t.shape[0]), flat_t] + 1e-9)
    loss_val = (per_token * flat_m).sum() / N

    loss = Tensor(_np.array(loss_val, dtype=_np.float32), _children=(logits,), _op="ce")
    loss.requires_grad = True
    loss.grad = xp.zeros_like(loss.data)

    def _backward():
        if logits.grad is not None:
            g = probs.copy()
            g_flat = g.reshape(-1, V)
            g_flat[_np.arange(flat_t.shape[0]), flat_t] -= 1.0
            g_flat *= flat_m[:, None]
            g_flat /= N
            logits.grad += from_numpy(g)

    loss._backward = _backward
    return loss


class ToolCallingLM(Module):
    """
    A tiny decoder-only Transformer for tool-call generation.

    forward(token_ids) → logits  (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=256,
    ):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_enc   = sinusoidal_encoding(max_seq_len, d_model)

        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        self.ln_final = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    @classmethod
    def from_config(cls, model_config):
        """Create a model from a config dict (the 'model' section)."""
        return cls(
            vocab_size=model_config["vocab_size"],
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            n_layers=model_config["n_layers"],
            d_ff=model_config["d_ff"],
            max_seq_len=model_config["max_seq_len"],
        )

    def __call__(self, token_ids):
        """
        Args:
            token_ids: integer array (B, T) — token IDs (numpy or device array).
        Returns:
            Tensor of logits, shape (B, T, vocab_size).
        """
        # Move token_ids to active device if needed
        token_ids = from_numpy(_np.asarray(token_ids, dtype=_np.int64))
        B, T = token_ids.shape

        x = self.token_emb(token_ids)           # (B, T, d)
        x = x + Tensor(self.pos_enc[:T])        # add positional encoding

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        return self.head(x)                     # (B, T, V)

    def generate(self, prompt_ids, max_new_tokens=64, eos_id=None):
        """
        Greedy auto-regressive generation.

        Args:
            prompt_ids: 1-D integer array of token IDs (the prompt).
            max_new_tokens: how many tokens to generate.
            eos_id: stop early if this token is produced.

        Returns:
            numpy ndarray of generated token IDs (prompt + new tokens).
        """
        ids = list(_np.asarray(to_numpy(prompt_ids), dtype=_np.int64))

        for _ in range(max_new_tokens):
            x = _np.array([ids], dtype=_np.int64)
            logits = self(x)                               # (1, current_len, V)
            logits_np = to_numpy(logits.data)
            next_id = int(logits_np[0, -1].argmax())      # greedy
            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return _np.array(ids, dtype=_np.int64)
