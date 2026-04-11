"""
Decoder-only Transformer for tool-calling.

Assembles the layers into a full language model and provides
a cross-entropy loss function and a simple greedy-generate method.
"""

import numpy as np
from .autograd import Tensor
from .layers import Module, Embedding, LayerNorm, Linear, TransformerBlock


def sinusoidal_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def cross_entropy_loss(logits, targets):
    """
    Args:
        logits:  Tensor of shape (B, T, V)  — raw model output
        targets: np.ndarray of shape (B, T)  — integer token IDs

    Returns:
        Scalar Tensor (the mean loss).
    """
    B, T, V = logits.shape

    # Numerically stable softmax
    shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)

    # Negative log-likelihood
    flat_p = probs.reshape(-1, V)
    flat_t = targets.reshape(-1)
    N = flat_t.shape[0]
    loss_val = -np.log(flat_p[np.arange(N), flat_t] + 1e-9).mean()

    loss = Tensor(np.array(loss_val), _children=(logits,), _op="ce")
    loss.requires_grad = True
    loss.grad = np.zeros_like(loss.data)

    def _backward():
        if logits.grad is not None:
            g = probs.copy()
            g_flat = g.reshape(-1, V)
            g_flat[np.arange(N), flat_t] -= 1.0
            g_flat /= N
            logits.grad += g  # loss.grad is 1.0 (scalar)

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
        self.pos_enc = sinusoidal_encoding(max_seq_len, d_model)

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
            token_ids: np.ndarray (B, T) of int token IDs.
        Returns:
            Tensor of logits, shape (B, T, vocab_size).
        """
        B, T = token_ids.shape

        x = self.token_emb(token_ids)        # (B, T, d)
        x = x + Tensor(self.pos_enc[:T])     # add positional encoding

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        return self.head(x)                  # (B, T, V)

    def generate(self, prompt_ids, max_new_tokens=64, eos_id=None):
        """
        Greedy auto-regressive generation.

        Args:
            prompt_ids: 1-D np.ndarray of int token IDs (the prompt).
            max_new_tokens: how many tokens to generate.
            eos_id: stop early if this token is produced.

        Returns:
            np.ndarray of generated token IDs (prompt + new tokens).
        """
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            x = np.array([ids], dtype=np.int64)        # (1, current_len)
            logits = self(x)                            # (1, current_len, V)
            next_id = int(logits.data[0, -1].argmax())  # greedy
            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return np.array(ids, dtype=np.int64)
