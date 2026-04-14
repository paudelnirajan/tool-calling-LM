"""
Backend abstraction for NumPy / CuPy / PyTorch / JAX.

Auto-detects the best available accelerator and exposes a NumPy-compatible
namespace as `xp`.  All other modules import from here instead of importing
numpy directly.

Usage:
    from backend import xp, to_numpy, from_numpy, device_name, set_backend

Detection priority (auto mode):
    1. CuPy        — NVIDIA GPU  (Colab GPU / any CUDA machine)
    2. PyTorch MPS — Apple Silicon GPU  (M1/M2/M3/M4)
    3. PyTorch CUDA— NVIDIA GPU via PyTorch (fallback if no CuPy)
    4. JAX         — Google TPU / GPU / CPU  (experimental)
    5. NumPy       — CPU fallback (always works, no install needed)

Installation:
    NVIDIA (Colab):   pip install cupy-cuda12x   # match your CUDA version
    Apple Silicon:    pip install torch           # MPS included
    Google TPU:       pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    NVIDIA via JAX:   pip install jax[cuda12]
"""

import numpy as _np

# ── Optional libraries — imported once, cached as module-level variables ──────
# The `# type: ignore` comments tell Pyright to skip resolution of these
# optional dependencies; they are intentionally absent from the project venv.

_torch = None
_cupy  = None
_jax   = None
_jnp   = None
_jr    = None   # jax.random

try:
    import cupy as _cupy          # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import torch as _torch        # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import jax as _jax            # type: ignore[import-untyped]
    import jax.numpy as _jnp      # type: ignore[import-untyped]
    import jax.random as _jr      # type: ignore[import-untyped]
except ImportError:
    pass

# ── Globals ───────────────────────────────────────────────────────────────────

_BACKEND: str = "numpy"   # "numpy" | "cupy" | "torch_mps" | "torch_cuda" | "jax"
xp = _np                  # active array module — numpy-compatible API


def device_name() -> str:
    """Return the active device: 'cpu', 'cuda', 'mps', or 'tpu/gpu/cpu'."""
    return {
        "numpy":      "cpu",
        "cupy":       "cuda",
        "torch_mps":  "mps",
        "torch_cuda": "cuda",
        "jax":        "tpu/gpu/cpu",
    }.get(_BACKEND, "cpu")


def to_numpy(arr) -> _np.ndarray:
    """Convert any backend array to a plain NumPy ndarray (for checkpoints etc.)."""
    if _BACKEND == "cupy" and _cupy is not None:
        if isinstance(arr, _cupy.ndarray):
            return arr.get()
    elif _BACKEND in ("torch_mps", "torch_cuda") and _torch is not None:
        if isinstance(arr, _torch.Tensor):
            return arr.detach().cpu().numpy()
    elif _BACKEND == "jax" and _jnp is not None:
        return _np.asarray(arr)
    return _np.asarray(arr)


def from_numpy(arr: _np.ndarray):
    """Convert a NumPy ndarray to the active backend array type."""
    arr = _np.asarray(arr)
    if _BACKEND == "cupy" and _cupy is not None:
        return _cupy.asarray(arr)
    elif _BACKEND in ("torch_mps", "torch_cuda") and _torch is not None:
        device = "mps" if _BACKEND == "torch_mps" else "cuda"
        if arr.dtype == _np.float32:
            return _torch.from_numpy(arr).to(device)
        return _torch.as_tensor(arr).to(device)
    elif _BACKEND == "jax" and _jnp is not None:
        return _jnp.array(arr)
    return arr


# ── set_backend ───────────────────────────────────────────────────────────────

def set_backend(name: str = "auto") -> None:
    """
    Choose the array backend.

    Args:
        name: 'auto' | 'cpu' | 'cuda' | 'mps' | 'tpu'
    """
    global xp, _BACKEND

    if name == "cpu":
        xp, _BACKEND = _np, "numpy"
        return

    if name in ("cuda", "gpu"):
        if _try_cupy():        return
        if _try_torch("cuda"): return
        raise RuntimeError(
            "No CUDA backend found. Install cupy-cuda12x or torch with CUDA support."
        )

    if name == "mps":
        if _try_torch("mps"):  return
        raise RuntimeError(
            "MPS not available. Requires PyTorch ≥1.12 on Apple Silicon (M1/M2/M3/M4)."
        )

    if name == "tpu":
        if _try_jax("tpu"):    return
        raise RuntimeError(
            "JAX TPU not available. "
            "Install: pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )

    if name == "auto":
        if _try_cupy():        return   # NVIDIA GPU via CuPy
        if _try_torch("mps"):  return   # Apple Silicon MPS
        if _try_torch("cuda"): return   # NVIDIA via PyTorch
        if _try_jax("auto"):   return   # JAX (TPU / GPU / CPU)
        xp, _BACKEND = _np, "numpy"    # CPU fallback
        return

    raise ValueError(f"Unknown backend {name!r}. Use 'auto', 'cpu', 'cuda', 'mps', or 'tpu'.")


# ── Backend probes ────────────────────────────────────────────────────────────

def _try_cupy() -> bool:
    global xp, _BACKEND
    if _cupy is None:
        return False
    try:
        _cupy.array([1.0])          # confirm device is accessible
        xp, _BACKEND = _cupy, "cupy"
        return True
    except Exception:
        return False


def _try_torch(device: str) -> bool:
    global xp, _BACKEND
    if _torch is None:
        return False
    try:
        if device == "mps":
            if not (hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available()):
                return False
            xp = _TorchBackend("mps")
            _BACKEND = "torch_mps"
        else:   # cuda
            if not _torch.cuda.is_available():
                return False
            xp = _TorchBackend("cuda")
            _BACKEND = "torch_cuda"
        return True
    except Exception:
        return False


def _try_jax(mode: str) -> bool:
    global xp, _BACKEND
    if _jax is None or _jnp is None:
        return False
    try:
        if mode == "tpu":
            devs = _jax.devices()
            if not any(d.platform == "tpu" for d in devs):
                return False
        xp = _JaxBackend()
        _BACKEND = "jax"
        return True
    except Exception:
        return False


# ── PyTorch shim ──────────────────────────────────────────────────────────────

class _TorchRandom:
    def __init__(self, device: str):
        self._device = device

    def randn(self, *shape):
        return _torch.randn(*shape, device=self._device)   # type: ignore[union-attr]

    def uniform(self, low, high, size):
        return _torch.empty(size, device=self._device).uniform_(float(low), float(high))   # type: ignore[union-attr]

    def shuffle(self, arr):
        """In-place shuffle — operates on NumPy index arrays."""
        _np.random.shuffle(arr)


class _TorchAdd:
    def __init__(self, device: str):
        self._device = device

    def at(self, target, indices, values):
        """Mimics np.add.at for embedding gradient scatter-add."""
        if _torch is not None and isinstance(target, _torch.Tensor):
            if not isinstance(indices, _torch.Tensor):
                indices = _torch.as_tensor(indices, dtype=_torch.long, device=self._device)
            if not isinstance(values, _torch.Tensor):
                values = _torch.as_tensor(_np.asarray(values), dtype=target.dtype, device=self._device)
            flat_idx = indices.reshape(-1)
            flat_val = values.reshape(flat_idx.shape[0], -1)
            target.scatter_add_(0, flat_idx.unsqueeze(-1).expand_as(flat_val), flat_val)
        else:
            _np.add.at(target, indices, values)


class _TorchBackend:
    """NumPy-compatible namespace backed by PyTorch (MPS or CUDA)."""

    def __init__(self, device: str):
        self._device = device
        self.random  = _TorchRandom(device)
        self.add     = _TorchAdd(device)
        self.float32 = _torch.float32    # type: ignore[union-attr]
        self.int64   = _torch.int64      # type: ignore[union-attr]
        self.pi      = float(_np.pi)

    # ── Array creation ────────────────────────────────────────────────────

    def asarray(self, data, dtype=None):
        return self.array(data, dtype=dtype)

    def array(self, data, dtype=None):
        dt = self._dt(dtype)
        if _torch is not None and isinstance(data, _torch.Tensor):
            return data.to(dtype=dt, device=self._device) if dt else data.to(device=self._device)
        na = _np.asarray(data)
        if dt:
            return _torch.as_tensor(na, dtype=dt, device=self._device)   # type: ignore[union-attr]
        return _torch.as_tensor(na, device=self._device)   # type: ignore[union-attr]

    def zeros(self, shape, dtype=None):
        return _torch.zeros(_s(shape), dtype=self._dt(dtype) or _torch.float32, device=self._device)   # type: ignore[union-attr]

    def ones(self, shape, dtype=None):
        return _torch.ones(_s(shape), dtype=self._dt(dtype) or _torch.float32, device=self._device)    # type: ignore[union-attr]

    def full(self, shape, fill_value, dtype=None):
        return _torch.full(_s(shape), fill_value, dtype=self._dt(dtype) or _torch.float32, device=self._device)   # type: ignore[union-attr]

    def zeros_like(self, a):
        if _torch is not None and isinstance(a, _torch.Tensor):
            return _torch.zeros_like(a)
        return _torch.zeros(a.shape, dtype=_torch.float32, device=self._device)   # type: ignore[union-attr]

    def ones_like(self, a):
        if _torch is not None and isinstance(a, _torch.Tensor):
            return _torch.ones_like(a)
        return _torch.ones(a.shape, dtype=_torch.float32, device=self._device)    # type: ignore[union-attr]

    def arange(self, *args):
        return _torch.arange(*args, device=self._device)   # type: ignore[union-attr]

    # ── Math ──────────────────────────────────────────────────────────────

    def exp(self, x):   return _torch.exp(self._t(x))    # type: ignore[union-attr]
    def log(self, x):   return _torch.log(self._t(x))    # type: ignore[union-attr]
    def sin(self, x):   return _torch.sin(self._t(x))    # type: ignore[union-attr]
    def cos(self, x):   return _torch.cos(self._t(x))    # type: ignore[union-attr]
    def tanh(self, x):  return _torch.tanh(self._t(x))   # type: ignore[union-attr]

    def sqrt(self, x):
        if isinstance(x, (int, float)):
            return float(_np.sqrt(x))
        return _torch.sqrt(self._t(x))   # type: ignore[union-attr]

    def sum(self, x, axis=None, keepdims=False):
        x = self._t(x)
        return x.sum() if axis is None else x.sum(dim=axis, keepdim=keepdims)

    def max(self, x, axis=None, keepdims=False):
        x = self._t(x)
        if axis is None:
            return x.max()
        return x.max(dim=axis, keepdim=keepdims).values

    def min(self, x, axis=None, keepdims=False):
        x = self._t(x)
        if axis is None:
            return x.min()
        return x.min(dim=axis, keepdim=keepdims).values

    def mean(self, x, axis=None, keepdims=False):
        x = self._t(x, dtype=_torch.float32)   # type: ignore[union-attr]
        return x.mean() if axis is None else x.mean(dim=axis, keepdim=keepdims)

    # ── Shape ops ─────────────────────────────────────────────────────────

    def swapaxes(self, x, a1, a2):       return self._t(x).transpose(a1, a2)
    def transpose(self, x, axes):         return self._t(x).permute(*axes)
    def broadcast_to(self, x, shape):     return self._t(x).expand(shape).clone()
    def expand_dims(self, x, axis):       return self._t(x).unsqueeze(axis)

    def triu(self, x, k=0):
        return _torch.triu(self._t(x), diagonal=k)   # type: ignore[union-attr]

    # ── Helpers ───────────────────────────────────────────────────────────

    def _t(self, x, dtype=None):
        """Ensure x is a torch.Tensor on the right device."""
        if _torch is not None and isinstance(x, _torch.Tensor):
            return x.to(device=self._device, dtype=dtype) if dtype else x.to(device=self._device)
        return _torch.as_tensor(_np.asarray(x), dtype=dtype, device=self._device)   # type: ignore[union-attr]

    def _dt(self, dtype):
        """Map numpy dtype → torch dtype."""
        if dtype is None:                                          return None
        if _torch is not None and dtype is _torch.float32:        return _torch.float32
        if _torch is not None and dtype is _torch.int64:          return _torch.int64
        if dtype == _np.float32:                                   return _torch.float32   # type: ignore[union-attr]
        if dtype == _np.int64:                                     return _torch.int64     # type: ignore[union-attr]
        return None


# ── JAX shim ──────────────────────────────────────────────────────────────────

class _JaxRandom:
    def __init__(self):
        self._key = _jr.PRNGKey(0) if _jr is not None else None   # type: ignore[union-attr]

    def _next_key(self):
        self._key, k = _jr.split(self._key)   # type: ignore[union-attr]
        return k

    def randn(self, *shape):
        return _jr.normal(self._next_key(), shape)   # type: ignore[union-attr]

    def uniform(self, low, high, size):
        return _jr.uniform(self._next_key(), size, minval=low, maxval=high)   # type: ignore[union-attr]

    def shuffle(self, arr):
        _np.random.shuffle(arr)


class _JaxAdd:
    def at(self, target, indices, values):
        _np.add.at(to_numpy(target), indices, to_numpy(values))


class _JaxBackend:
    """NumPy-compatible namespace backed by jax.numpy."""

    def __init__(self):
        self.random  = _JaxRandom()
        self.add     = _JaxAdd()
        self.float32 = _np.float32
        self.int64   = _np.int64
        self.pi      = float(_np.pi)

    def asarray(self, x, dtype=None):  return _jnp.asarray(x, dtype=dtype)   # type: ignore[union-attr]
    def array(self, x, dtype=None):    return _jnp.array(x, dtype=dtype)      # type: ignore[union-attr]

    def zeros(self, shape, dtype=None):
        return _jnp.zeros(_s(shape), dtype=dtype or _np.float32)   # type: ignore[union-attr]

    def ones(self, shape, dtype=None):
        return _jnp.ones(_s(shape), dtype=dtype or _np.float32)    # type: ignore[union-attr]

    def full(self, shape, fill_value, dtype=None):
        return _jnp.full(_s(shape), fill_value, dtype=dtype or _np.float32)   # type: ignore[union-attr]

    def zeros_like(self, a):  return _jnp.zeros_like(a)   # type: ignore[union-attr]
    def ones_like(self, a):   return _jnp.ones_like(a)    # type: ignore[union-attr]
    def arange(self, *args): return _jnp.arange(*args)   # type: ignore[union-attr]

    def exp(self, x):   return _jnp.exp(x)    # type: ignore[union-attr]
    def log(self, x):   return _jnp.log(x)    # type: ignore[union-attr]
    def sin(self, x):   return _jnp.sin(x)    # type: ignore[union-attr]
    def cos(self, x):   return _jnp.cos(x)    # type: ignore[union-attr]
    def tanh(self, x):  return _jnp.tanh(x)   # type: ignore[union-attr]

    def sqrt(self, x):
        if isinstance(x, (int, float)): return float(_np.sqrt(x))
        return _jnp.sqrt(x)   # type: ignore[union-attr]

    def sum(self, x, axis=None, keepdims=False):
        return _jnp.sum(x, axis=axis, keepdims=keepdims)   # type: ignore[union-attr]

    def max(self, x, axis=None, keepdims=False):
        return _jnp.max(x, axis=axis, keepdims=keepdims)   # type: ignore[union-attr]

    def min(self, x, axis=None, keepdims=False):
        return _jnp.min(x, axis=axis, keepdims=keepdims)   # type: ignore[union-attr]

    def mean(self, x, axis=None, keepdims=False):
        return _jnp.mean(x, axis=axis, keepdims=keepdims)  # type: ignore[union-attr]

    def swapaxes(self, x, a1, a2):       return _jnp.swapaxes(x, a1, a2)     # type: ignore[union-attr]
    def transpose(self, x, axes):         return _jnp.transpose(x, axes)       # type: ignore[union-attr]
    def broadcast_to(self, x, shape):     return _jnp.broadcast_to(x, shape)   # type: ignore[union-attr]
    def expand_dims(self, x, axis):       return _jnp.expand_dims(x, axis)     # type: ignore[union-attr]
    def triu(self, x, k=0):               return _jnp.triu(x, k)               # type: ignore[union-attr]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _s(shape):
    """Normalise a shape argument to a tuple."""
    return (shape,) if isinstance(shape, int) else tuple(shape)


# ── Auto-detect on import ─────────────────────────────────────────────────────
set_backend("auto")


def print_backend_info():
    """Print which backend is active."""
    name = xp.__name__ if hasattr(xp, "__name__") else type(xp).__name__
    print(f"Backend: {_BACKEND}  |  device: {device_name()}  |  xp: {name}")


if __name__ == "__main__":
    print_backend_info()
