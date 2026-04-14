"""Adam optimizer — device-agnostic via the backend abstraction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as _np
from backend import xp


class Adam:
    """
    Adam optimizer (Kingma & Ba, 2014).

    Moments are stored on the same device as the parameters so the update
    step runs entirely on the accelerator.

    Args:
        params: list of Tensor objects (from model.parameters()).
        lr: peak learning rate.
        betas: (beta1, beta2) for moment estimates.
        eps: numerical stability term.
        grad_clip: if > 0, clip gradient global norm to this value.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, grad_clip=1.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.grad_clip = grad_clip
        self.t = 0

        # First and second moment estimates — live on the active device
        self.m = [xp.zeros_like(p.data) for p in params]
        self.v = [xp.zeros_like(p.data) for p in params]

    def step(self):
        """Update all parameters using their .grad."""
        self.t += 1

        # Optional gradient clipping (global norm)
        if self.grad_clip > 0:
            total_norm = xp.sqrt(
                sum(xp.sum(p.grad ** 2) for p in self.params if p.grad is not None)
            )
            # scalar comparison works for numpy, cupy, and torch tensors
            if float(total_norm) > self.grad_clip:
                scale = self.grad_clip / (float(total_norm) + 1e-9)
                for p in self.params:
                    if p.grad is not None:
                        p.grad *= scale

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update weights (in-place on device)
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)


class CosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.lr

    def step(self, current_step):
        if current_step < self.warmup_steps:
            lr = self.base_lr * (current_step + 1) / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.base_lr * 0.5 * (1 + _np.cos(_np.pi * progress))

        self.optimizer.lr = lr
        return lr
