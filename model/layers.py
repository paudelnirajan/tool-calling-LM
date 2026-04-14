"""
Neural-network layers built on the autograd engine — device-agnostic.

Every layer is a Module whose __call__ builds the computation graph
and whose parameters() returns the learnable Tensors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import xp
from .autograd import Tensor


class Module:
    """Recursively discovers parameters in all child Modules / Tensors."""

    def parameters(self):
        params = []
        for v in vars(self).values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Embedding(Module):
    def __init__(self, num_embeddings, dim):
        self.weight = Tensor(
            xp.asarray(xp.random.randn(num_embeddings, dim), dtype=xp.float32) * 0.02,
            requires_grad=True,
        )

    def __call__(self, indices):
        """indices: array of ints, shape (B, T)."""
        out_data = self.weight.data[indices]
        out = Tensor(out_data, _children=(self.weight,), _op="embed")

        w = self.weight  # capture for closure

        def _backward():
            if w.grad is not None:
                # xp.add.at handles the scatter-add for all backends
                xp.add.at(w.grad, indices, out.grad)

        out._backward = _backward
        return out


class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        limit = xp.sqrt(6.0 / (in_dim + out_dim))  # Xavier
        self.weight = Tensor(
            xp.asarray(xp.random.uniform(-float(limit), float(limit), (in_dim, out_dim)), dtype=xp.float32),
            requires_grad=True,
        )
        self.bias = (
            Tensor(xp.zeros(out_dim, dtype=xp.float32), requires_grad=True)
            if bias
            else None
        )

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.gain = Tensor(xp.ones(dim, dtype=xp.float32), requires_grad=True)
        self.bias = Tensor(xp.zeros(dim, dtype=xp.float32), requires_grad=True)
        self.eps = eps

    def __call__(self, x):
        mu = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        std_inv = 1.0 / xp.sqrt(var + self.eps)
        x_hat = (x.data - mu) * std_inv

        out_data = self.gain.data * x_hat + self.bias.data
        out = Tensor(out_data, _children=(x, self.gain, self.bias), _op="ln")

        gain, bias, eps = self.gain, self.bias, self.eps
        D = x.data.shape[-1]

        def _backward():
            if gain.grad is not None:
                gain.grad += (out.grad * x_hat).reshape(-1, D).sum(axis=0)
            if bias.grad is not None:
                bias.grad += out.grad.reshape(-1, D).sum(axis=0)
            if x.grad is not None:
                dx_hat = out.grad * gain.data
                x.grad += (
                    std_inv
                    / D
                    * (
                        D * dx_hat
                        - dx_hat.sum(axis=-1, keepdims=True)
                        - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
                    )
                )

        out._backward = _backward
        return out


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = d_model // n_heads

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)

    def __call__(self, x):
        B, T, D = x.shape
        H, dk = self.n_heads, self.dk

        # Project
        Q = self.W_q(x).reshape(B, T, H, dk).transpose(0, 2, 1, 3)  # (B,H,T,dk)
        K = self.W_k(x).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        V = self.W_v(x).reshape(B, T, H, dk).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        K_t = K.transpose(0, 1, 3, 2)                      # (B,H,dk,T)
        scores = (Q @ K_t) * (1.0 / xp.sqrt(dk))

        # Causal mask (upper triangle → −∞)
        mask = xp.triu(xp.ones((T, T), dtype=xp.float32), k=1) * (-1e9)
        scores = scores + Tensor(mask)

        weights = scores.softmax(axis=-1)                   # (B,H,T,T)

        # Weighted sum → concat heads → output projection
        attn = (weights @ V)                                # (B,H,T,dk)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.W_o(attn)


class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def __call__(self, x):
        return self.fc2(self.fc1(x).gelu())


class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))   # residual around attention
        x = x + self.ffn(self.ln2(x))    # residual around FFN
        return x
