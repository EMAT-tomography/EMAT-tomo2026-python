"""Ports of the cSAXS (PSI) MATLAB math/utility routines used by the alignment
procedures (vertical mass fluctuation and tomographic self-consistency).

All functions work on stacks laid out as ``(n_items, n_rows, n_cols)`` (the
``rgtomo`` convention) unless stated otherwise.

*-----------------------------------------------------------------------*
|                                                                       |
|  Except where otherwise noted, this work is licensed under a          |
|  Creative Commons Attribution-NonCommercial-ShareAlike 4.0            |
|  International (CC BY-NC-SA 4.0) license.                             |
|                                                                       |
|  Copyright (c) 2017 by Paul Scherrer Institute (http://www.psi.ch)    |
|                                                                       |
|       Author: CXS group, PSI                                          |
*-----------------------------------------------------------------------*
You may use this code with the following provisions:

If the code is fully or partially redistributed, or rewritten in another
  computing language this notice should be included in the redistribution.

If this code, or subfunctions or parts of it, is used for research in a
  publication or if it is fully or partially rewritten for another
  computing language the authors and institution should be acknowledged
  in written form in the publication: "Data processing was carried out
  using the "cSAXS matlab package" developed by the CXS group,
  Paul Scherrer Institut, Switzerland."
  Variations on the latter text can be incorporated upon discussion with
  the CXS group if needed to more specifically reflect the use of the package
  for the published work.

A publication that focuses on describing features, or parameters, that
   are already existing in the code should be first discussed with the
   authors.

This code and subroutines are part of a continuous development, they
   are provided "as they are" without guarantees or liability on part
   of PSI or the authors. It is the user responsibility to ensure its
   proper use and the correctness of the results.

Python port: R. Girod (UAntwerp), 2026.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from ._matlab import conv_same, gausswin, matlab_round

_WORKERS = -1  # use all cores for FFTs


def _fft(x, axis):
    return sfft.fft(x, axis=axis, workers=_WORKERS)


def _ifft(x, axis):
    return sfft.ifft(x, axis=axis, workers=_WORKERS)


def _fft2(x):
    return sfft.fft2(x, axes=(-2, -1), workers=_WORKERS)


def _ifft2(x):
    return sfft.ifft2(x, axes=(-2, -1), workers=_WORKERS)


def _bshape(ndim, axis, n):
    """Shape broadcasting a length-n vector along ``axis``."""
    shape = [1] * ndim
    shape[axis] = n
    return shape


def _single(a) -> bool:
    return np.asarray(a).dtype in (np.float32, np.complex64, np.float16)


def _cdtype(a):
    """Complex dtype matching the precision of ``a``."""
    return np.complex64 if _single(a) else np.complex128


def _rdtype(a):
    """Real dtype matching the precision of ``a``."""
    return np.float32 if _single(a) else np.float64


def _per_item(values, ndim, n_items):
    """Broadcast per-item (axis 0) values against an ndim array."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 0:
        return v
    return v.reshape([n_items] + [1] * (ndim - 1))


# --------------------------------------------------------------------------- shifting
def fft_shift_axis(img, shift, axis, apply_fft=True):
    """Sub-pixel shift along ``axis`` with the Fourier shift theorem
    (port of ``imshift_fft_ax``). Positive shifts move content towards higher
    indices. ``shift`` is a scalar or one value per item along axis 0."""
    img = np.asarray(img)
    axis = axis % img.ndim
    shift = np.asarray(shift, dtype=float)
    if np.all(shift == 0):
        return img
    is_real = np.isrealobj(img)
    n = img.shape[axis]
    grid = np.fft.fftfreq(n).reshape(_bshape(img.ndim, axis, n))
    s = _per_item(shift, img.ndim, img.shape[0])
    factor = np.exp(-2j * np.pi * s * grid).astype(_cdtype(img))
    if apply_fft:
        out = _ifft(_fft(img, axis) * factor, axis)
    else:
        out = img * factor
    if is_real and apply_fft:
        out = out.real.astype(_rdtype(img), copy=False)
    return out


def _shift_factor_2d(ny, nx, x, y, n_items=None, dtype=np.complex128):
    xg = np.fft.fftfreq(nx)
    yg = np.fft.fftfreq(ny)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 0 and y.ndim == 0:
        return np.exp(-2j * np.pi * (x * xg[None, :] + y * yg[:, None])).astype(dtype)
    x = np.broadcast_to(x, (n_items,)).reshape(-1, 1, 1)
    y = np.broadcast_to(y, (n_items,)).reshape(-1, 1, 1)
    return np.exp(-2j * np.pi * (x * xg[None, None, :] + y * yg[None, :, None])).astype(dtype)


def imshift_fft(img, x, y=None, apply_fft=True):
    """Port of ``imshift_fft``: shift 2D images by (x, y) pixels (x along
    columns, y along rows) with periodic boundary conditions.

    ``img`` is (ny, nx) or (n, ny, nx); ``x``/``y`` scalars or per-image vectors.
    If ``y`` is None, ``x`` is an (n, 2) array of [x, y] shifts.
    """
    img = np.asarray(img)
    if y is None:
        xy = np.asarray(x, dtype=float).reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(x == 0) and np.all(y == 0):
        return img
    if apply_fft and np.all(x == 0):
        return fft_shift_axis(img, y, axis=-2) if img.ndim == 3 or y.ndim == 0 else \
            fft_shift_axis(img[None], y, axis=-2)[0]
    if apply_fft and np.all(y == 0):
        return fft_shift_axis(img, x, axis=-1) if img.ndim == 3 or x.ndim == 0 else \
            fft_shift_axis(img[None], x, axis=-1)[0]
    is_real = np.isrealobj(img)
    ny, nx = img.shape[-2:]
    n_items = img.shape[0] if img.ndim == 3 else None
    factor = _shift_factor_2d(ny, nx, x, y, n_items=n_items, dtype=_cdtype(img))
    if apply_fft:
        out = _ifft2(_fft2(img) * factor)
        if is_real:
            out = out.real.astype(_rdtype(img), copy=False)
        return out
    return img * factor


def imshift_linear(img, x, y=None, fill=0.0):
    """Port of ``imshift_linear``: bilinear shift (content moves by +x, +y),
    values outside the field of view are set to ``fill``."""
    from scipy import ndimage as ndi
    img = np.asarray(img)
    if y is None:
        xy = np.asarray(x, dtype=float).reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
    x = np.broadcast_to(np.asarray(x, dtype=float), (img.shape[0],))
    y = np.broadcast_to(np.asarray(y, dtype=float), (img.shape[0],))
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        out[i] = ndi.shift(img[i], (y[i], x[i]), order=1, mode="constant", cval=fill)
    return out


# --------------------------------------------------------------------------- edges / filters
def smooth_edges(img, win_size=5, axes=(-2, -1)):
    """Port of ``smooth_edges``: smooth the periodic boundary (last ``win_size``
    and first ``win_size`` samples) along the given axes to avoid FFT edge jumps."""
    img = np.array(img, copy=True)
    for ax in np.atleast_1d(axes):
        ax = int(ax) % img.ndim
        n = img.shape[ax]
        if n <= 2 * win_size:
            continue
        win_size = max(win_size, 3)
        ind = np.concatenate([np.arange(n - win_size, n), np.arange(win_size)])
        tmp = np.take(img, ind, axis=ax)
        kernel = gausswin(win_size, 2.5)
        tmp = conv_same(tmp, kernel, axis=ax)
        norm = conv_same(np.ones(ind.size), kernel)
        tmp = tmp / norm.reshape(_bshape(img.ndim, ax, ind.size))
        idx = [slice(None)] * img.ndim
        idx[ax] = ind
        img[tuple(idx)] = tmp.astype(img.dtype, copy=False)
    return img


def imfilter_high_pass_1d(img, axis, sigma, padding=0, apply_fft=True):
    """Port of ``imfilter_high_pass_1d``: Fourier high-pass filter
    ``exp(-sigma^2 / f^2)`` along one axis (sigma scaled to be resolution
    independent). ``sigma == 0`` applies a derivative filter instead."""
    img = np.asarray(img)
    axis = axis % img.ndim
    padding = int(np.ceil(padding))
    if padding > 0:
        pad = [(0, 0)] * img.ndim
        pad[axis] = (padding, padding)
        img = np.pad(img, pad, mode="symmetric")
    n = img.shape[axis]
    is_real = np.isrealobj(img)
    if apply_fft:
        img = _fft(img, axis)
    x = (np.arange(n) - n / 2) / n
    sigma_s = 256.0 / (n - 2 * padding) * sigma
    if sigma_s == 0:
        filt = (2j * np.pi * (np.fft.fftshift(np.arange(n) / n) - 0.5)).astype(_cdtype(img))
    else:
        with np.errstate(divide="ignore", over="ignore"):
            filt = np.fft.fftshift(np.exp(1.0 / (-(x ** 2) / sigma_s ** 2))).astype(_rdtype(img))
    img = img * filt.reshape(_bshape(img.ndim, axis, n))
    if apply_fft:
        img = _ifft(img, axis)
    if is_real:
        img = img.real
    if padding > 0:
        img = img[_axis_slice(img.ndim, axis, slice(padding, n - padding))]
    return img


def _axis_slice(ndim, axis, sl):
    idx = [slice(None)] * ndim
    idx[axis] = sl
    return tuple(idx)


# --------------------------------------------------------------------------- phase helpers
def get_phase_gradient_1d(img, axis=-1, step=0.5, shift=0.0):
    """Port of ``get_phase_gradient_1D``. Real inputs are interpreted as phase
    (``exp(1j*img)``); returns the phase derivative along ``axis``."""
    img = np.asarray(img)
    axis = axis % img.ndim
    if np.isrealobj(img):
        img = np.exp(1j * img.astype(np.float32)).astype(np.complex64)
    pad = 8
    padw = [(0, 0)] * img.ndim
    padw[axis] = (pad, pad)
    img = np.pad(img, padw, mode="symmetric")
    img = smooth_edges(img, pad, axes=[axis])
    if step == 0:
        img = img / (np.abs(img) + np.finfo(np.float32).eps)
        n = img.shape[axis]
        X = 2j * np.pi * np.fft.fftfreq(n)
        d = _ifft(_fft(img, axis) * X.reshape(_bshape(img.ndim, axis, n)), axis)
        d_img = np.imag(np.conj(img) * d)
    else:
        a = fft_shift_axis(img, -step + shift, axis)
        b = fft_shift_axis(img, step + shift, axis)
        d_img = np.angle(a * np.conj(b)) / (2 * step)
    # MATLAB removes indices pad:end-pad-1 (1-based), i.e. a one pixel offset
    # that is compensated in get_img_int_1d
    size = d_img.shape[axis]
    d_img = d_img[_axis_slice(d_img.ndim, axis, slice(pad - 1, size - pad - 1))]
    return d_img.astype(np.float32, copy=False)


def get_img_int_1d(grad, axis):
    """Port of ``get_img_int_1D``: Fourier integration along one axis (including
    the original one-pixel shift that compensates ``get_phase_gradient_1d``)."""
    grad = np.asarray(grad)
    axis = axis % grad.ndim
    n = grad.shape[axis]
    f = np.fft.fftfreq(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.exp(2j * np.pi * f) / (2j * np.pi * f)
    X[0] = 0
    X = X.astype(_cdtype(grad)).reshape(_bshape(grad.ndim, axis, n))
    return _ifft(_fft(grad, axis) * X, axis)


def remove_sinogram_ramp(sinogram, air_gap, polyfit_order=-1):
    """Port of ``remove_sinogram_ramp`` for (n, n_layers, width) stacks: subtract
    a linear ramp (along x) through the mean values of the left/right air gaps."""
    s = np.asarray(sinogram)
    g = np.ceil(np.atleast_1d(np.asarray(air_gap, dtype=float))).astype(int)
    g1, g2 = g[0], g[min(g.size - 1, 1)]
    W = s.shape[-1]
    vals = [s[..., :g1].mean(axis=-1, keepdims=True),
            s[..., W - g2 - 1:].mean(axis=-1, keepdims=True)]
    if polyfit_order == 0:
        vals = [v.mean(axis=-2, keepdims=True) for v in vals]
    elif polyfit_order == 1:
        from ._matlab import mad0
        nl = s.shape[-2]
        ramp = np.linspace(-1, 1, nl).reshape([1] * (s.ndim - 2) + [nl, 1])
        fitted = []
        for v in vals:
            weight = np.ones_like(v)
            for _ in range(10):
                plane = (np.mean(weight * v, axis=-2, keepdims=True) / np.mean(weight, axis=-2, keepdims=True)
                         + np.mean(weight * v * ramp, axis=-2, keepdims=True)
                         / np.mean(weight * ramp ** 2, axis=-2, keepdims=True) * ramp)
                deviation = 5 * mad0(v - plane)
                weight = 1.0 / (1 + (np.abs(v - plane) / deviation) ** 2)
            fitted.append(plane)
        vals = fitted
    x = (np.arange(1, W + 1) / W).astype(s.dtype if np.issubdtype(s.dtype, np.floating) else float)
    ramp = vals[0] + (vals[1] - vals[0]) * x
    return (s - ramp).astype(s.dtype if np.issubdtype(s.dtype, np.floating) else float, copy=False)


def unwrap2d_fft(phase_diff, axis=-1, empty_region=None):
    """Port of ``unwrap2D_fft`` for real phase-derivative inputs."""
    phase = get_img_int_1d(phase_diff, axis).real
    if empty_region is not None and len(np.atleast_1d(empty_region)) and \
            (axis % np.ndim(phase_diff)) == np.ndim(phase_diff) - 1:
        phase = remove_sinogram_ramp(phase, empty_region, -1)
    return phase


# --------------------------------------------------------------------------- resampling
def imgaussfilt3_conv(x, filter_size):
    """Port of ``imgaussfilt3_conv``: separable Gaussian convolution with kernel
    ``exp(-(k/s)^2)`` truncated at ``ceil(2 s)`` and zero padding.
    ``filter_size`` gives one value per axis (0 = no filtering)."""
    x = np.asarray(x)
    fs = np.broadcast_to(np.asarray(filter_size, dtype=float), (x.ndim,))
    for ax in range(x.ndim):
        s = fs[ax]
        if s == 0:
            continue
        grid = np.arange(-np.ceil(2 * s), np.ceil(2 * s) + 1) / s
        ker = np.exp(-grid ** 2)
        ker /= ker.sum()
        x = conv_same(x, ker, axis=ax)
    return x


def crop_pad(img, outsize):
    """Port of ``crop_pad``: centred crop or zero padding of the last two axes
    (centre at ``floor(N/2)``)."""
    img = np.asarray(img)
    nin = img.shape[-2:]
    nout = tuple(int(v) for v in outsize[:2])
    if nin == nout:
        return img
    out = np.zeros(img.shape[:-2] + nout, dtype=img.dtype)
    off = [nout[i] // 2 - nin[i] // 2 for i in range(2)]
    src, dst = [], []
    for i in range(2):
        d0 = max(off[i], 0)
        d1 = min(off[i] + nin[i], nout[i])
        s0 = max(-off[i], 0)
        s1 = min(-off[i] + nout[i], nin[i])
        dst.append(slice(d0, d1))
        src.append(slice(s0, s1))
    out[..., dst[0], dst[1]] = img[..., src[0], src[1]]
    return out


def interpolate_ft_centered(img, new_size, interp_sign=1):
    """Port of ``interpolateFT_centered``: Fourier down/up-sampling of the last
    two axes with a +/- 0.5 px shift that keeps the pixel centres aligned."""
    img = np.asarray(img)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[None]
    ny, nx = img.shape[-2:]
    np_new = (int(new_size[0]) + 2, int(new_size[1]) + 2)
    is_real = np.isrealobj(img)
    scale = (np_new[0] - 2) * (np_new[1] - 2) / (ny * nx)
    downsample = int(np.ceil(np.sqrt(1.0 / scale)))
    img = np.pad(img, [(0, 0), (downsample, downsample), (downsample, downsample)], mode="symmetric")
    cd = _cdtype(img)
    F = _fft2(img)
    F = F * _shift_factor_2d(F.shape[-2], F.shape[-1], -0.5 * interp_sign, -0.5 * interp_sign, dtype=cd)
    F = np.fft.ifftshift(crop_pad(np.fft.fftshift(F, axes=(-2, -1)), np_new), axes=(-2, -1))
    F = F * _shift_factor_2d(np_new[0], np_new[1], 0.5 * interp_sign, 0.5 * interp_sign, dtype=cd)
    out = _ifft2(F) * np.asarray(scale, dtype=_rdtype(img))
    out = out[:, 1:-1, 1:-1]
    if is_real:
        out = out.real.astype(_rdtype(img), copy=False)
    return out[0] if squeeze else out


def interpolate_linear(img, new_size):
    """Port of ``interpolate_linear``: bilinear resampling of the last two axes
    onto ``linspace(1, N, new)`` grids."""
    img = np.asarray(img)
    out = img
    for ax, nn in zip((-2, -1), new_size[:2]):
        n = out.shape[ax]
        nn = int(nn)
        if n == nn:
            continue
        p = np.linspace(0, n - 1, nn)
        if n == 1:
            out = np.repeat(out, nn, axis=ax)
            continue
        i0 = np.clip(np.floor(p).astype(int), 0, n - 2)
        w = (p - i0).reshape(_bshape(out.ndim, ax % out.ndim, nn))
        a = np.take(out, i0, axis=ax)
        b = np.take(out, i0 + 1, axis=ax)
        out = a * (1 - w) + b * w
    return out.astype(img.dtype if np.issubdtype(img.dtype, np.floating) else np.float64, copy=False)


# --------------------------------------------------------------------------- rotation
def imrotate_ax_fft(img, theta):
    """Port of ``imrotate_ax_fft`` (axis 3): rotate each image of a stack by
    ``theta`` degrees (scalar or per image) using three Fourier shears."""
    img = np.asarray(img)
    theta = np.asarray(theta, dtype=float)
    if np.all(theta == 0) or img.size == 0:
        return img
    squeeze = img.ndim == 2
    if squeeze:
        img = img[None]
    is_real = np.isrealobj(img)
    n_items = img.shape[0]
    theta = np.broadcast_to(theta, (n_items,)).copy()
    off = matlab_round(theta / 90).astype(int)
    if np.any(off != 0):
        img = img.copy()
        for i in np.flatnonzero(off):
            img[i] = np.rot90(img[i], off[i])
        theta = theta - 90 * off
    M, N = img.shape[-2:]
    xgrid = np.fft.fftfreq(M)[None, :, None]        # along rows
    ygrid = np.fft.fftfreq(N)[None, None, :]        # along columns
    Mgrid = (np.arange(1, M + 1) - np.floor(M / 2) - 0.5)[None, :, None]
    Ngrid = (np.arange(1, N + 1) - np.floor(N / 2) - 0.5)[None, None, :]
    th = theta.reshape(-1, 1, 1)
    Nx = -np.sin(np.deg2rad(th)) * xgrid
    Ny = np.tan(np.deg2rad(th) / 2) * ygrid
    cd = _cdtype(img)
    M1 = np.exp(-2j * np.pi * Mgrid * Ny).astype(cd)
    M2 = np.exp(-2j * np.pi * Ngrid * Nx).astype(cd)
    out = _ifft(_fft(img, -1) * M1, -1)
    out = _ifft(_fft(out, -2) * M2, -2)
    out = _ifft(_fft(out, -1) * M1, -1)
    if is_real:
        out = out.real.astype(_rdtype(img), copy=False)
    return out[0] if squeeze else out


# --------------------------------------------------------------------------- gradients
def get_img_grad(img):
    """Port of ``get_img_grad``: Fourier gradients (dX along columns, dY along rows)."""
    img = np.asarray(img)
    is_real = np.isrealobj(img)
    nx, ny = img.shape[-1], img.shape[-2]
    X = (2j * np.pi * np.fft.fftfreq(nx)).astype(_cdtype(img))
    Y = (2j * np.pi * np.fft.fftfreq(ny)).astype(_cdtype(img))[:, None]
    dX = _ifft(_fft(img, -1) * X, -1)
    dY = _ifft(_fft(img, -2) * Y, -2)
    if is_real:
        dX, dY = dX.real, dY.real
    return dX, dY


def _grad_conv_kernel(win_size):
    N = max(9, 2 * win_size + 1)
    grid = 2j * np.pi * (np.fft.fftshift(np.arange(N) / N) - 0.5)
    ker = -np.real(np.fft.fftshift(np.fft.fft(grid))) / N
    c = int(np.ceil(N / 2)) - 1
    r = int(np.ceil(win_size))
    return ker[c - r:c + r + 1].astype(np.float32)


def get_img_grad_conv(img, win_size, axis):
    """Port of ``get_img_grad_conv``: derivative along ``axis`` by convolution
    with a truncated spectral-derivative kernel (avoids FFT edge issues)."""
    return conv_same(np.asarray(img), _grad_conv_kernel(win_size), axis=axis)


# --------------------------------------------------------------------------- masks
def radtap(X, Y, tappix, zerorad):
    """Port of ``radtap``: radial cosine taper (0 inside ``zerorad``, 1 outside
    ``zerorad + tappix``)."""
    tau = 2 * tappix
    R = np.sqrt(X ** 2 + Y ** 2)
    taper = 0.5 * (1 + np.cos(2 * np.pi * (R - zerorad - tau / 2) / tau))
    taper = (R > zerorad + tau / 2) * 1.0 + taper * (R <= zerorad + tau / 2)
    return taper * (R >= zerorad)


def circular_mask(npix, radial_smooth=None, rad_apod=0.0):
    """Circular apodization mask of ``apply_3D_apodization`` ("circulo"):
    1 in the centre, cosine roll-off of width ``radial_smooth`` to 0 at the edge."""
    if radial_smooth is None:
        radial_smooth = npix / 50
    radial_smooth = max(radial_smooth, 1)
    xt = np.arange(npix) - npix / 2
    X, Y = np.meshgrid(xt, xt)
    return (1 - radtap(X, Y, radial_smooth, matlab_round(npix / 2 - rad_apod - radial_smooth))).astype(np.float32)


def center_of_mass(X):
    """Centre of mass of each image of a stack (``center.m`` behaviour with
    ``use_shift=true``): returns (pos_x, pos_y, mass), positions relative to
    the image centre so that a uniform image gives (0, 0)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[None]
    N, M = X.shape[-2:]
    mass = X.sum(axis=(-2, -1))
    xgrid = np.arange(1, M + 1)
    ygrid = np.arange(1, N + 1)
    pos_x = (X.sum(axis=-2) * xgrid).sum(axis=-1) / mass - M / 2 - 0.5
    pos_y = (X.sum(axis=-1) * ygrid).sum(axis=-1) / mass - N / 2 - 0.5
    return pos_x, pos_y, mass
