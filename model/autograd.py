"""
Lightweight autograd engine for NumPy tensors.

Tracks operations in a computation graph and computes gradients
via reverse-mode backpropagation (just like PyTorch, but tiny).
"""

import numpy as np

def _sum_to_shape(x, shape):
    """Sum *x* over any axes that were broadcast to reach *shape*."""
    # Remove extra leading dims  (e.g. (B,T,d) -> (d,))
    while x.ndim > len(shape):
        x = x.sum(axis=0)
    # Collapse dims that are 1 in the target  (e.g. (B,1,d) kept as-is)
    for i, s in enumerate(shape):
        if s == 1 and x.shape[i] != 1:
            x = x.sum(axis=i, keepdims=True)
    return x


class Tensor:
    """A NumPy array with automatic differentiation."""

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float32)

        # Auto-propagate requires_grad from parents
        if not requires_grad and _children:
            requires_grad = any(
                c.requires_grad for c in _children if isinstance(c, Tensor)
            )

        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad={self.requires_grad})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=np.float32))
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            if self.grad is not None:
                self.grad += _sum_to_shape(out.grad, self.shape)
            if other.grad is not None:
                other.grad += _sum_to_shape(out.grad, other.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        # scalar
        if isinstance(other, (int, float)):
            out = Tensor(self.data * other, _children=(self,), _op="*s")

            def _backward():
                if self.grad is not None:
                    self.grad += out.grad * other

            out._backward = _backward
            return out

        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=np.float32))

        out = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            if self.grad is not None:
                self.grad += _sum_to_shape(out.grad * other.data, self.shape)
            if other.grad is not None:
                other.grad += _sum_to_shape(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        """Matrix multiply with full broadcast support."""
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward():
            if self.grad is not None:
                g = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += _sum_to_shape(g, self.shape)
            if other.grad is not None:
                g = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _sum_to_shape(g, other.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op="reshape")

        def _backward():
            if self.grad is not None:
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """Permute dimensions.  e.g. t.transpose(0, 2, 1, 3)"""
        out = Tensor(np.transpose(self.data, axes), _children=(self,), _op="T")

        def _backward():
            if self.grad is not None:
                inv = [0] * len(axes)
                for i, a in enumerate(axes):
                    inv[a] = i
                self.grad += np.transpose(out.grad, inv)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, _children=(self,), _op="softmax")

        def _backward():
            if self.grad is not None:
                dot = (s * out.grad).sum(axis=axis, keepdims=True)
                self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    def gelu(self):
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        y = 0.5 * x * (1.0 + t)
        out = Tensor(y, _children=(self,), _op="gelu")

        def _backward():
            if self.grad is not None:
                sech2 = 1.0 - t * t
                dx = 0.5 * (1.0 + t) + 0.5 * x * sech2 * c * (
                    1.0 + 3.0 * 0.044715 * x * x
                )
                self.grad += out.grad * dx

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op="sum",
        )

        def _backward():
            if self.grad is not None:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad += np.broadcast_to(g, self.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def backward(self):
        """Run reverse-mode autodiff from this (scalar) tensor."""
        # Topological sort
        order, visited = [], set()

        def _topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    _topo(child)
                order.append(t)

        _topo(self)

        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            t._backward()

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.0)
