"""Alignment of tilt series.

- :func:`align_stack_xcorr` and helpers: port of ``align_stack_xcorr``,
  ``find_shifts_2d_xcorr`` and ``apply_shifts_2d`` (tomohawk, with R. Girod's
  windowing, edge filtering, slab stretching and ROI options).
- :func:`align_stack_features`: port of ``align_stack_features`` (feature
  tracking + RANSAC). MATLAB uses SURF blobs; SURF is not available in
  scikit-image, so the closest equivalent (SIFT blobs) is used.
- :func:`csaxs_vmf`: port of ``rg_cSAXS_VMF`` (vertical mass fluctuation /
  common line refinement, adapted from the cSAXS package, see the licence
  notice in :mod:`rgtomo.csaxs`).
- :func:`apply_tilt_axis`: the tilt-axis correction applied by
  ``align_tilt_axis_gui``.
- :func:`align_proj_com`: port of ``wh_align_proj_com``.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

from ._matlab import hamming, matlab_round, quantile, tukeywin
from .csaxs import (crop_pad, fft_shift_axis, get_img_grad_conv,
                    imfilter_high_pass_1d)
from .preprocessing import find_edges_stack, gaussfilter_slice
from .series import TiltSeries

__all__ = [
    "dftregistration", "find_shifts_2d_xcorr", "apply_shifts_2d", "align_stack_xcorr",
    "align_stack_features", "csaxs_vmf", "apply_tilt_axis", "align_proj_com",
]

_W = -1


# =========================================================================== xcorr
def _argmax_colmajor(a):
    """Index of the first maximum in MATLAB (column-major) order."""
    idx = int(np.argmax(a.T))
    c, r = np.unravel_index(idx, a.T.shape)
    return int(r), int(c)


def _dftups(inp, nor, noc, usfac, roff, coff):
    nr, nc = inp.shape
    kernc = np.exp((-2j * np.pi / (nc * usfac))
                   * (np.fft.ifftshift(np.arange(nc))[:, None] - np.floor(nc / 2))
                   * (np.arange(noc)[None, :] - coff))
    kernr = np.exp((-2j * np.pi / (nr * usfac))
                   * (np.arange(nor)[:, None] - roff)
                   * (np.fft.ifftshift(np.arange(nr))[None, :] - np.floor(nr / 2)))
    return kernr @ inp @ kernc


def _ftpad(imft, outsize):
    nin = imft.shape
    out = crop_pad(np.fft.fftshift(imft), outsize)
    return np.fft.ifftshift(out) * (outsize[0] * outsize[1]) / (nin[0] * nin[1])


def dftregistration(buf1ft, buf2ft, usfac=1):
    """Efficient sub-pixel image registration by cross-correlation.

    Port of ``dftregistration`` (as embedded in ``find_shifts_2d_xcorr.m``).
    ``buf1ft``/``buf2ft`` are the FFTs of the reference image and of the image
    to register. Returns (row_shift, col_shift, diffphase, CCmax).

    Copyright (c) 2016, Manuel Guizar Sicairos, James R. Fienup, University of
    Rochester. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution
    * Neither the name of the University of Rochester nor the names
      of its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Citation: M. Guizar-Sicairos, S. T. Thurman and J. R. Fienup, "Efficient
    subpixel image registration algorithms", Opt. Lett. 33, 156-158 (2008).
    """
    nr, nc = buf2ft.shape
    if usfac == 1:
        CC = sfft.ifft2(buf1ft * np.conj(buf2ft), workers=_W)
        r, c = _argmax_colmajor(np.abs(CC))
        CCmax = CC[r, c]
        row_shift, col_shift = r, c
        if row_shift > np.floor(nr / 2):
            row_shift -= nr
        if col_shift > np.floor(nc / 2):
            col_shift -= nc
    elif usfac > 1:
        CC = sfft.ifft2(_ftpad(buf1ft * np.conj(buf2ft), (2 * nr, 2 * nc)), workers=_W)
        r, c = _argmax_colmajor(np.abs(CC))
        CCmax = CC[r, c]
        Nr2 = np.fft.ifftshift(np.arange(-nr, nr))
        Nc2 = np.fft.ifftshift(np.arange(-nc, nc))
        row_shift = Nr2[r] / 2
        col_shift = Nc2[c] / 2
        if usfac > 2:
            row_shift = matlab_round(row_shift * usfac) / usfac
            col_shift = matlab_round(col_shift * usfac) / usfac
            dftshift = np.fix(np.ceil(usfac * 1.5) / 2)
            n_ups = int(np.ceil(usfac * 1.5))
            CC = np.conj(_dftups(buf2ft * np.conj(buf1ft), n_ups, n_ups, usfac,
                                 dftshift - row_shift * usfac, dftshift - col_shift * usfac))
            rloc, cloc = _argmax_colmajor(np.abs(CC))
            # the matrix-multiply DFT is not normalized like ifft2: rescale so that
            # CCmax has the same meaning for every upsampling factor (the MATLAB
            # code returned an N times larger correlation coefficient here)
            CCmax = CC[rloc, cloc] / (nr * nc)
            row_shift = row_shift + (rloc - dftshift) / usfac
            col_shift = col_shift + (cloc - dftshift) / usfac
    else:
        raise ValueError("usfac must be >= 1")
    return float(row_shift), float(col_shift), float(np.angle(CCmax)), float(np.real(CCmax))


def find_shifts_2d_xcorr(target, reference=None, upsampling=1, verbose=True):
    """Find the shifts of a stack of images by FFT cross-correlation
    (``find_shifts_2d_xcorr``).

    Without ``reference`` the images are aligned in cascade (each image to the
    previous one) and shifts are accumulated. Returns ``shifts`` ``(n, 2)``
    [row, column] and the Pearson correlation coefficient of each pair."""
    from tqdm.auto import tqdm
    if isinstance(target, TiltSeries):
        target = target.data
    if isinstance(reference, TiltSeries):
        reference = reference.data
    target = np.asarray(target, dtype=np.float32)
    n, n_rows, n_cols = target.shape
    shifts = np.zeros((n, 2))
    corr = np.ones(n)
    if verbose:
        print("Finding shifts via cross-correlation...")
    ft = sfft.fft2(target, axes=(1, 2), workers=_W)
    if reference is None:
        for i in tqdm(range(1, n), desc="xcorr", leave=False, disable=not verbose):
            rs, cs, _, cc = dftregistration(ft[i], ft[i - 1], upsampling)
            shifts[i] = np.array([rs, cs]) + shifts[i - 1]
            corr[i] = cc
        means = target.mean(axis=(1, 2), dtype=np.float64)
        stds = target.std(axis=(1, 2), ddof=1, dtype=np.float64)
        corr[1:] = corr[1:] / (n_rows * n_cols)
        corr[1:] = (corr[1:] - means[1:] * means[:-1]) / (stds[1:] * stds[:-1])
    else:
        reference = np.asarray(reference, dtype=np.float32)
        fr = sfft.fft2(reference, axes=(1, 2), workers=_W)
        for i in tqdm(range(n), desc="xcorr", leave=False, disable=not verbose):
            rs, cs, _, cc = dftregistration(ft[i], fr[i], upsampling)
            shifts[i] = [rs, cs]
            corr[i] = cc
        corr = corr / (n_rows * n_cols)
        mt = target.mean(axis=(1, 2), dtype=np.float64)
        mr = reference.mean(axis=(1, 2), dtype=np.float64)
        st = target.std(axis=(1, 2), ddof=1, dtype=np.float64)
        sr = reference.std(axis=(1, 2), ddof=1, dtype=np.float64)
        corr = (corr - mt * mr) / (st * sr)
    return shifts, corr


def apply_shifts_2d(stack, shifts, method="linear", border="pad"):
    """Shift each image by ``shifts[i] = [row, col]`` pixels (``apply_shifts_2d``).

    ``border='pad'`` replicates the edge pixels, ``'black'`` fills with zeros."""
    if isinstance(stack, TiltSeries):
        return stack.with_data(apply_shifts_2d(stack.data, shifts, method, border))
    order = {"nearest": 0, "linear": 1, "bilinear": 1, "cubic": 3, "bicubic": 3}[method]
    mode = "nearest" if border == "pad" else "constant"
    s = np.asarray(stack, dtype=np.float32)
    out = np.empty_like(s)
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    for i in range(s.shape[0]):
        if np.all(shifts[i] == 0):
            out[i] = s[i]
        else:
            out[i] = ndi.shift(s[i], shifts[i], order=order, mode=mode, cval=0.0)
    return out


def _stretch(stack, angles):
    """Stretch each projection by 1/cos(angle) along x about the image centre
    (slab-like geometry), linear interpolation, zero fill."""
    out = np.empty_like(stack)
    cx = (stack.shape[2] - 1) / 2
    for i in range(stack.shape[0]):
        c = np.cos(np.deg2rad(angles[i]))
        out[i] = ndi.affine_transform(stack[i], [1.0, c], offset=[0.0, cx - c * cx],
                                      order=1, mode="constant", cval=0.0)
    return out


def align_stack_xcorr(series, reference_slice=None, window=False, edge=0, iterations=1,
                      slab_like=False, upsampling=1, y_range=None, verbose=True):
    """Align a tilt series by FFT cross-correlation (``align_stack_xcorr``).

    Parameters
    ----------
    series : TiltSeries
    reference_slice : int or None
        Index (0-based) of the projection all images are aligned to; ``None``
        aligns every image to its neighbour (cascade).
    window : bool
        Apply a Hamming window before correlating (helps if the ROI is not
        well isolated).
    edge : float
        If > 0, correlate the Sobel gradient magnitude of the images filtered
        with a Gaussian of this sigma (edge detector).
    iterations : int
        Number of alignment iterations.
    slab_like : bool
        Stretch the images by 1/cos(angle) for the correlation (continuous
        slab-like samples with a vertical tilt axis).
    upsampling : int
        Sub-pixel precision 1/upsampling.
    y_range : (start, stop) or None
        Rows (0-based, stop excluded) used to compute the shifts.

    Returns
    -------
    aligned : ndarray
        Aligned stack (shifts applied with linear interpolation, replicated edges).
    shifts : ndarray (n, 2)
        [row, col] shifts of the last iteration.
    corr_coefs : ndarray (n,)
    target_stack_find_shifts : ndarray
        The (filtered/windowed) images used to find the shifts, aligned.
    """
    target = np.array(series.data, dtype=np.float32, copy=True)
    angles = series.angles
    shifts = corr = tf = None
    for it in range(iterations):
        if verbose:
            print(f"xcorr alignment, iteration {it + 1}")
        tf = target[:, y_range[0]:y_range[1], :] if y_range is not None else target
        L, l = tf.shape[1], tf.shape[2]
        if slab_like:
            if verbose:
                print("Stretching for slab-like geometry...")
            tf = _stretch(tf, angles)
        if window:
            if verbose:
                print("Applying a hamming window...")
            tf = tf * (hamming(L)[:, None] * hamming(l)[None, :]).astype(np.float32)
        elif verbose:
            print("Aligning without windowing...")
        if edge > 0:
            if verbose:
                print("Applying an edge detector...")
            tf = find_edges_stack(tf, edge)
        elif verbose:
            print("Aligning with regular images...")
        ref = None
        if reference_slice is not None:
            ref = np.repeat(tf[reference_slice][None], tf.shape[0], axis=0)
        shifts, corr = find_shifts_2d_xcorr(tf, ref, upsampling, verbose=verbose)
        target = apply_shifts_2d(target, -shifts, "linear", "pad")
        tf = apply_shifts_2d(tf, -shifts)
    return target, shifts, corr, tf


# =========================================================================== feature tracking
def _ransac_translation(src, dst, max_distance, max_trials, rng):
    """MSAC estimate of a pure translation mapping ``src`` onto ``dst``."""
    d = dst - src
    m = d.shape[0]
    cand = np.arange(m) if m <= max_trials else rng.choice(m, max_trials, replace=False)
    best_cost, best_inl = np.inf, None
    t2 = max_distance ** 2
    for i in cand:
        r2 = np.sum((d - d[i]) ** 2, axis=1)
        cost = np.minimum(r2, t2).sum()
        if cost < best_cost:
            best_cost, best_inl = cost, r2 < t2
    t = d[best_inl].mean(axis=0)
    # final refit with the inliers of the refined model
    inl = np.sum((d - t) ** 2, axis=1) < t2
    if inl.any():
        t = d[inl].mean(axis=0)
    return t, inl


def _show_matches(ref, moving, p1, p2, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    rgb = np.stack([moving / (moving.max() or 1), ref / (ref.max() or 1), moving / (moving.max() or 1)], -1)
    ax.imshow(np.clip(rgb, 0, 1))
    ax.plot(p1[:, 1], p1[:, 0], "o", mfc="none", mec="r", ms=5, label="reference")
    ax.plot(p2[:, 1], p2[:, 0], "+", color="g", ms=6, label="moving")
    for a, b in zip(p1, p2):
        ax.plot([a[1], b[1]], [a[0], b[0]], "y-", lw=0.8)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    plt.show()


def align_stack_features(stack, c_dog=0.005, max_ratio=0.6, transform="translation",
                         iterations=3, max_distance=3.0, max_trials=2000,
                         gaussian_sigma=1.0, pad_fraction=0.2, upsampling=1, show=True, seed=0):
    """Feature-based alignment (port of ``align_stack_features``).

    Blob-like features are detected on Gaussian-filtered images (SIFT here,
    SURF in MATLAB), matched between consecutive projections, and the
    transform between them is estimated robustly (RANSAC/MSAC). The transform
    is applied to the raw images; each image is registered to the previous,
    already registered one. Repeated ``iterations`` times.

    Parameters
    ----------
    stack : ndarray or TiltSeries
    c_dog : float
        SIFT contrast threshold (on images normalized to [0, 1] individually);
        decrease to detect more blobs (smooth isolated particles may need
        0.001-0.002), increase to keep only the strongest ones.
    max_ratio : float
        Lowe ratio test for descriptor matching (MATLAB ``MaxRatio``).
    transform : {'translation', 'euclidean', 'similarity', 'affine'}
    max_distance : float
        Inlier distance (px) for RANSAC.
    gaussian_sigma : float
        Pre-filter for the feature detection.
    upsampling : int
        SIFT image upsampling (1 = fast, 2 = more features).
    show : bool
        Plot all and inlier matches for the first pair.
    """
    from skimage.feature import SIFT, match_descriptors
    if isinstance(stack, TiltSeries):
        return stack.with_data(align_stack_features(
            stack.data, c_dog, max_ratio, transform, iterations, max_distance, max_trials,
            gaussian_sigma, pad_fraction, upsampling, show, seed))
    rng = np.random.default_rng(seed)
    data = np.array(stack, dtype=np.float32, copy=True)
    n = data.shape[0]
    padsize = int(np.ceil(data.shape[1] * pad_fraction))
    pad = [(padsize, padsize), (padsize, padsize)]
    crop = (slice(padsize, -padsize), slice(padsize, -padsize))
    sift = SIFT(upsampling=upsampling, c_dog=c_dog)

    def features(img):
        try:
            sift.detect_and_extract(img)
            return sift.keypoints.astype(float), sift.descriptors
        except RuntimeError:
            return np.zeros((0, 2)), np.zeros((0, 128), dtype=np.uint8)

    def warp(img, tf_model, t):
        if t is not None:
            return ndi.shift(img, t, order=1, mode="constant", cval=0.0)
        from skimage.transform import warp as skwarp
        return skwarp(img, tf_model.inverse, order=1, mode="constant", cval=0.0,
                      preserve_range=True).astype(np.float32)

    min_matches = 2 if transform == "translation" else 3
    for it in range(iterations):
        # features are detected on Gaussian-filtered images normalized to [0, 1]
        # individually; the transforms are applied to the raw images
        g = np.asarray(gaussfilter_slice(data, gaussian_sigma), dtype=np.float32)
        lo = g.min(axis=(1, 2), keepdims=True)
        hi = g.max(axis=(1, 2), keepdims=True)
        g = (g - lo) / np.where(hi > lo, hi - lo, 1)
        print(f"Iteration {it + 1}/{iterations}: registering {n - 1} images with {transform} transform")
        kp_prev = des_prev = None
        for k in range(n - 1):
            moving = np.pad(g[k + 1], pad, mode="edge")
            ref = np.pad(g[k], pad, mode="edge")
            ali = np.pad(data[k + 1], pad, mode="edge")
            if kp_prev is None:
                kp_prev, des_prev = features(ref)
            kp1, d1 = kp_prev, des_prev
            kp2, d2 = features(moving)
            if len(kp1) == 0 or len(kp2) == 0:
                print(f"  image {k + 1}: no features found, left unchanged")
                kp_prev, des_prev = kp2, d2
                continue
            matches = match_descriptors(d1, d2, max_ratio=max_ratio, cross_check=True)
            if len(matches) < min_matches:
                print(f"  image {k + 1}: only {len(matches)} matches, left unchanged")
                kp_prev, des_prev = kp2, d2
                continue
            m1, m2 = kp1[matches[:, 0]], kp2[matches[:, 1]]
            if show and it == 0 and k == 0:
                _show_matches(ref, moving, m1, m2, f"All matched features ({len(m1)})")
            if transform == "translation":
                t, inl = _ransac_translation(m2, m1, max_distance, max_trials, rng)
                model = None
            else:
                from skimage.measure import ransac
                from skimage import transform as sktf
                cls = {"euclidean": sktf.EuclideanTransform, "rigid": sktf.EuclideanTransform,
                       "similarity": sktf.SimilarityTransform, "affine": sktf.AffineTransform}[transform]
                model, inl = ransac((m2[:, ::-1], m1[:, ::-1]), cls, min_samples=3,
                                    residual_threshold=max_distance, max_trials=max_trials, rng=seed)
                t = None
                if model is None:
                    print(f"  image {k + 1}: RANSAC failed, left unchanged")
                    kp_prev, des_prev = kp2, d2
                    continue
            if show and it == 0 and k == 0:
                _show_matches(ref, moving, m1[inl], m2[inl], f"Inlier matches ({int(inl.sum())})")
            data[k + 1] = warp(ali, model, t)[crop]
            moved = warp(moving, model, t)
            g[k + 1] = moved[crop]
            # features of the registered image are the reference for the next pair
            if t is not None:
                kp_prev, des_prev = kp2 + t, d2
            else:
                kp_prev, des_prev = features(moved)
    return data


# =========================================================================== VMF
def _remove_linear_ramp(a):
    """Subtract a linear ramp along the layer axis (axis 1) through the mean of
    the 5 top and 6 bottom layers (``remove_linear_ramp`` of rg_cSAXS_VMF)."""
    L = a.shape[1]
    top = a[:, :5].mean(axis=1, keepdims=True)
    bottom = a[:, -6:].mean(axis=1, keepdims=True)
    x = np.arange(1, L + 1)[None, :] / L
    return a - (top + (bottom - top) * x)


def _shift_nearest(a, shift, sign=1.0):
    """Nearest-neighbour shift along axis 1 with zero fill (``interp1``/``interp2``
    'nearest' with 0 extrapolation). Samples ``a[n, j - sign*shift[n]]``."""
    n, L = a.shape
    q = np.arange(L)[None, :] - sign * np.asarray(shift, dtype=float)[:, None]
    valid = (q >= 0) & (q <= L - 1)
    idx = np.clip(matlab_round(q).astype(int), 0, L - 1)
    out = np.take_along_axis(a, idx, axis=1)
    return np.where(valid, out, 0).astype(a.dtype)


def _fill_gaps_1d(invar0, mask, n_modes, n_iter):
    if not np.any(mask):
        return invar0
    invar = invar0.copy()
    lo, hi = quantile(invar[~mask], [0.01, 0.99])
    for _ in range(n_iter):
        U, S, Vt = np.linalg.svd(invar, full_matrices=False)
        filt = (U[:, :n_modes] * S[:n_modes]) @ Vt[:n_modes]
        invar = np.where(mask, filt, invar)
        invar = np.clip(invar, lo, hi)
    return invar


def _linear_iterative_refinement(invar_0, weights_0, data_filter):
    """Optical-flow refinement of the vertical shifts (layout (n_angles, n_layers))."""
    n_angles, L = invar_0.shape
    total = np.zeros(n_angles, dtype=np.float64)
    invar_0 = _remove_linear_ramp(invar_0)
    invar_0 = imfilter_high_pass_1d(invar_0, 1, data_filter, L / 2)
    invar_0 = invar_0 * tukeywin(L, 0.1)[None, :]
    invar_0 = _fill_gaps_1d(invar_0, weights_0 == 0, 2, 20)
    relax_step = 0.9
    err = []
    win = tukeywin(L, 0.2)[None, :]
    invar = invar_0
    weights = weights_0
    for i in range(1000):
        invar = fft_shift_axis(invar_0, total, axis=1)
        # (the original code samples the weights at j + shift, i.e. the opposite direction)
        weights = _shift_nearest(weights_0, total, sign=-1.0)
        invar = imfilter_high_pass_1d(invar, 1, data_filter, L / 2)
        invar = invar * win
        m_invar = (invar * weights).sum(axis=0) / (weights.sum(axis=0) + 1e-3)
        md_invar = get_img_grad_conv(m_invar, 2, axis=0)
        DY = m_invar[None, :] - invar
        shift = -np.sum(weights * DY * md_invar[None, :], axis=1) / np.sum(weights * md_invar[None, :] ** 2, axis=1)
        shift = relax_step * np.minimum(0.5, np.abs(shift)) * np.sign(shift)
        total = total + shift
        err.append(float(np.mean(weights * DY ** 2)))
        if (i > 1 and err[i - 1] < err[i]) or np.max(np.abs(shift)) < 1e-2:
            break
    return total, invar, weights


def csaxs_vmf(series, data_filter=0.01, plot=True, outer_loops_refinement=3):
    """Vertical alignment by vertical mass fluctuation / common line
    (port of ``rg_cSAXS_VMF``, adapted from cSAXS ``align_tomo_initial``).

    Each projection is summed along x; if the tilt axis is well aligned these
    vertical profiles are identical for all angles, so the vertical shifts are
    found by minimizing their differences (high-pass filtered, optical-flow
    refinement).

    Parameters
    ----------
    series : TiltSeries
    data_filter : float
        High-pass filter (lower = more filtering), 0.005-0.01 for STEM.
    plot : bool
        Show the invariant before/after alignment and the applied shifts.

    Returns
    -------
    TiltSeries with vertically shifted projections (shifts in ``meta['vmf_shift']``).
    """
    stack = np.asarray(series.data, dtype=np.float32)
    angles = np.asarray(series.angles, dtype=float)
    n_angles, n_layers, _ = stack.shape
    invar0 = stack.sum(axis=2, dtype=np.float64).astype(np.float32)   # (n_angles, n_layers)
    weight = np.ones_like(invar0)
    invar = _remove_linear_ramp(invar0)
    # select range without boundary issues (MATLAB round(2:Nlayers-1))
    invar = invar[:, 1:n_layers - 1]
    weight = weight[:, 1:n_layers - 1]
    L = invar.shape[1]
    if L < 20:
        raise ValueError("Too small range for vertical alignment")
    invar = _remove_linear_ramp(invar)
    invar_init = imfilter_high_pass_1d(invar, 1, data_filter, L / 2)

    shift_y = np.zeros(n_angles)
    invar_shifted = weights_shifted = None
    rng_idx = np.arange(L)
    from tqdm.auto import tqdm
    for _ in tqdm(range(outer_loops_refinement), desc="VMF refinement", leave=False):
        inv_s = fft_shift_axis(invar, shift_y, axis=1)
        w_s = _shift_nearest(weight, shift_y, sign=1.0)
        offset = np.max(np.abs(shift_y))
        rng_idx = matlab_round(np.arange(1 + offset, L - offset + 1e-9, 1.0)).astype(int) - 1
        if rng_idx.size <= 30:
            raise ValueError("Too small vertical range for alignment")
        inv_s, w_s = inv_s[:, rng_idx], w_s[:, rng_idx]
        upd, invar_shifted, weights_shifted = _linear_iterative_refinement(inv_s, w_s, data_filter)
        shift_y = shift_y + upd

    optimal_shift = shift_y - np.median(shift_y)
    out = series.with_data(fft_shift_axis(stack, optimal_shift, axis=1).astype(np.float32))
    out.meta["vmf_shift"] = optimal_shift

    if plot:
        _plot_vmf(angles, invar_init, weight, invar_shifted, weights_shifted, rng_idx, shift_y)
    return out


def _plot_vmf(angles, invar_init, weight, invar_shifted, weights_shifted, rng_idx, shift_y):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    L = invar_init.shape[1]
    ext = [angles.min(), angles.max(), 1, L]
    order = np.argsort(angles)
    clim = quantile(invar_init[np.isfinite(invar_init)], [1e-2, 1 - 1e-2])
    axs[0, 0].imshow(invar_init[order].T, aspect="auto", origin="lower", extent=ext,
                     vmin=clim[0], vmax=clim[1], cmap="viridis")
    axs[0, 0].set(title="No alignment, linear ramp removed", ylabel="Vertical axis [pixels]",
                  xlabel="Angles [deg]")
    axs[0, 1].plot(np.where(weight > 0, invar_init, np.nan).T, lw=0.6)
    axs[0, 1].set(xlabel="Vertical pixels", title="Line plot - highpass filtered")
    axs[0, 2].imshow((1 - weight)[order].T, aspect="auto", origin="lower", extent=ext, cmap="gray")
    axs[0, 2].set(title="Phase jumps / Residues mask", ylabel="Vertical axis [pixels]", xlabel="Angles [deg]")
    if invar_shifted is not None:
        clim = quantile(invar_shifted, [1e-2, 1 - 1e-2])
        axs[1, 0].imshow(invar_shifted[order].T, aspect="auto", origin="lower",
                         extent=[angles.min(), angles.max(), rng_idx[0] + 1, rng_idx[-1] + 1],
                         vmin=clim[0], vmax=clim[1], cmap="viridis")
        axs[1, 0].set(title="Iterative refinement", xlabel="Angles [deg]", ylabel="Vertical axis [pixels]")
        axs[1, 1].plot(np.where(weights_shifted > 0, invar_shifted, np.nan).T, lw=0.6)
        axs[1, 1].set(xlabel="Vertical pixels", title="Line plot - highpass filtered")
    axs[1, 2].plot(angles[order], shift_y[order], ".-")
    axs[1, 2].set(title="Applied shift", ylabel="Shift [pixels]", xlabel="Angles [deg]")
    axs[1, 2].grid(True)
    fig.suptitle("Tomography invariant vertical alignment")
    fig.tight_layout()
    plt.show()


# =========================================================================== tilt axis / COM
def apply_tilt_axis(series, translation=0.0, rotation=0.0, order=1):
    """Apply a tilt-axis correction (the transform of ``align_tilt_axis_gui``).

    Parameters
    ----------
    translation : float
        Tilt axis translation in pixels; positive moves the axis to the right
        (the images are shifted to the left).
    rotation : float
        Tilt axis rotation in degrees; positive corresponds to a clockwise
        rotation of the axis (the images are rotated counter-clockwise).

    Edges are replicated (MATLAB pads with replicated values before rotating).
    """
    if translation == 0 and rotation == 0:
        return series.copy() if isinstance(series, TiltSeries) else np.array(series, copy=True)
    data = series.data if isinstance(series, TiltSeries) else np.asarray(series)
    data = np.asarray(data, dtype=np.float32)
    ny, nx = data.shape[-2:]
    th = np.deg2rad(rotation)
    rinv = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    ctr = np.array([(ny - 1) / 2, (nx - 1) / 2])
    offset = rinv @ (np.array([0.0, translation]) - ctr) + ctr
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        out[i] = ndi.affine_transform(data[i], rinv, offset=offset, order=order, mode="nearest")
    if isinstance(series, TiltSeries):
        res = series.with_data(out)
        res.meta["tilt_axis"] = {"translation": translation, "rotation": rotation}
        return res
    return out


def align_proj_com(stack):
    """Translate the whole stack so that the centre of mass of the summed
    projections is at the image centre (``wh_align_proj_com``; integer
    circular shift)."""
    s = stack.data if isinstance(stack, TiltSeries) else np.asarray(stack)
    overlay = s.sum(axis=0, dtype=np.float64)
    mass = overlay.sum()
    ny, nx = overlay.shape
    x, y = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1))
    dx = (x * overlay).sum() / mass - nx / 2
    dy = (y * overlay).sum() / mass - ny / 2
    sy, sx = (int(v) for v in matlab_round(-np.array([dy, dx])))
    out = np.roll(s, (sy, sx), axis=(1, 2))
    return stack.with_data(out) if isinstance(stack, TiltSeries) else out
