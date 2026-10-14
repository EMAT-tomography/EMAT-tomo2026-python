"""Small helpers reproducing MATLAB semantics that differ from NumPy/SciPy defaults.

Keeping these in one place makes the ports of the MATLAB toolbox behave like
the original code (rounding, quantiles, window definitions, 'same'
convolution alignment, imgaussfilt filter size, Otsu threshold, ...).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import windows as _win


def matlab_round(x):
    """MATLAB ``round``: halves are rounded away from zero (NumPy rounds to even)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def iround(x) -> int:
    """MATLAB-rounded scalar as Python int."""
    return int(matlab_round(x))


def quantile(x, q, axis=None):
    """MATLAB ``quantile``/``prctile`` interpolation (Hazen, p_k = (k - 0.5)/n)."""
    return np.quantile(np.asarray(x), q, axis=axis, method="hazen")


def prctile(x, p, axis=None):
    return quantile(x, np.asarray(p) / 100.0, axis=axis)


def mad0(x):
    """MATLAB ``mad(x, 0)``: mean absolute deviation."""
    x = np.asarray(x).ravel()
    return np.mean(np.abs(x - np.mean(x)))


# --------------------------------------------------------------------------- windows
def tukeywin(n: int, r: float = 0.5) -> np.ndarray:
    """MATLAB ``tukeywin(n, r)`` (symmetric)."""
    if n == 1:
        return np.ones(1)
    return _win.tukey(n, alpha=r, sym=True)


def gausswin(n: int, a: float = 2.5) -> np.ndarray:
    """MATLAB ``gausswin(n, a)``."""
    if n == 1:
        return np.ones(1)
    N = n - 1
    k = np.arange(n) - N / 2
    return np.exp(-0.5 * (a * k / (N / 2)) ** 2)


def hanning(n: int) -> np.ndarray:
    """MATLAB ``hanning(n)`` (no zero end points)."""
    k = np.arange(1, n + 1)
    return 0.5 * (1 - np.cos(2 * np.pi * k / (n + 1)))


def hamming(n: int) -> np.ndarray:
    """MATLAB ``hamming(n)`` (symmetric)."""
    if n == 1:
        return np.ones(1)
    return _win.hamming(n, sym=True)


# --------------------------------------------------------------------------- convolution
def _axis_slice(ndim, axis, sl):
    idx = [slice(None)] * ndim
    idx[axis] = sl
    return tuple(idx)


def conv_same(a, k, axis=-1):
    """1D convolution along ``axis`` with MATLAB ``conv(..., 'same')`` alignment
    and zero padding (correct for even-length kernels too)."""
    a = np.asarray(a)
    k = np.asarray(k).ravel()
    axis = axis % a.ndim
    n = a.shape[axis]
    nk = k.size
    out = np.zeros(a.shape, dtype=np.result_type(a.dtype, k.dtype))
    c = nk // 2
    for j in range(nk):
        s = c - j  # out[i] += k[j] * a[i + s]
        if abs(s) >= n:
            continue
        if s >= 0:
            src, dst = slice(s, n), slice(0, n - s)
        else:
            src, dst = slice(0, n + s), slice(-s, n)
        out[_axis_slice(a.ndim, axis, dst)] += k[j] * a[_axis_slice(a.ndim, axis, src)]
    return out


# --------------------------------------------------------------------------- filtering
def imgaussfilt(img, sigma, axes=None):
    """MATLAB ``imgaussfilt``/``imgaussfilt3``: Gaussian filter with filter size
    ``2*ceil(2*sigma)+1`` and replicate padding.

    ``axes`` restricts filtering to some axes (e.g. ``(1, 2)`` to filter each
    projection of a stack independently)."""
    if np.isscalar(sigma) and sigma == 0:
        return img
    img = np.asarray(img)
    out_dtype = img.dtype if np.issubdtype(img.dtype, np.floating) else np.float64
    if axes is None:
        axes = tuple(range(img.ndim))
    sig = np.zeros(img.ndim)
    rad = np.zeros(img.ndim, dtype=int)
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (len(axes),))
    for ax, si in zip(axes, s):
        sig[ax] = si
        rad[ax] = int(np.ceil(2 * si))
    return ndi.gaussian_filter(img.astype(out_dtype, copy=False), sigma=tuple(sig),
                               radius=tuple(rad), mode="nearest")


def imgradient_sobel(img):
    """MATLAB ``imgradient(img, 'sobel')`` gradient magnitude (replicate borders)."""
    img = np.asarray(img, dtype=np.float64 if img.dtype == np.float64 else np.float32)
    gx = ndi.sobel(img, axis=-1, mode="nearest")
    gy = ndi.sobel(img, axis=-2, mode="nearest")
    return np.hypot(gx, gy)


# --------------------------------------------------------------------------- intensities
def rescale(a, lower=0.0, upper=1.0):
    """MATLAB ``rescale``: linear mapping of the global [min, max] to [lower, upper].
    A constant input returns ``lower``."""
    a = np.asarray(a)
    out_dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64
    a = a.astype(out_dtype, copy=False)
    amin, amax = np.nanmin(a), np.nanmax(a)
    if amax == amin:
        return np.full(a.shape, lower, dtype=out_dtype)
    return ((a - amin) / (amax - amin) * (upper - lower) + lower).astype(out_dtype, copy=False)


def im2double(a):
    """MATLAB ``im2double``: integers are scaled by their dtype range."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64)
    if a.dtype == np.bool_:
        return a.astype(np.float64)
    info = np.iinfo(a.dtype)
    return (a.astype(np.float64) - info.min) / (info.max - info.min)


def graythresh(img) -> float:
    """MATLAB ``graythresh``: Otsu threshold on a 256-bin histogram of the image
    converted with ``im2uint8`` (floating point data is clipped to [0, 1])."""
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.floating):
        u8 = np.round(np.clip(np.nan_to_num(a), 0, 1) * 255).astype(np.uint8)
    elif a.dtype == np.uint8:
        u8 = a
    else:
        u8 = np.round(im2double(a) * 255).astype(np.uint8)
    counts = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    p = counts / counts.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, 257))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    maxval = np.nanmax(sigma_b2) if np.any(np.isfinite(sigma_b2)) else np.nan
    if not np.isfinite(maxval):
        return 0.0
    idx = np.mean(np.flatnonzero(sigma_b2 == maxval)) + 1  # 1-based like MATLAB
    return float((idx - 1) / 255)
