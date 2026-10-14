"""Removal of distorted projections.

- :func:`generate_clean_tilt_series`: port of ``generateCleanTiltSeries`` and
  its helpers ``createRefStack``, ``computeSSIM``, ``removeBadImages`` and
  ``generateCleanSeries`` (FastAutomatedTomography, A. Kadu et al., UAntwerp).
  The series is reconstructed (EM, 5 iterations), re-projected, and each
  projection is compared to its re-projection with SSIM; the projections that
  best follow a robust polynomial fit of the SSIM are kept.
- :func:`generate_clean_series`: port of ``rg_generate_clean_series`` (error to
  the neighbouring frames, W. Heyvaert).
- :func:`select_images`: manual exclusion list (``select_images_gui_2``).
- :func:`fasttomo_keepone`, :func:`sharpness`.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ._matlab import imgradient_sobel, rescale
from .alignment import align_proj_com
from .reconstruction import fp, reconstruct
from .series import TiltSeries

__all__ = ["generate_clean_tilt_series", "generate_clean_series", "select_images",
           "fasttomo_keepone", "sharpness", "ssim", "robustfit", "mse"]


# =========================================================================== metrics
def ssim(a, b, data_range=1.0, sigma=1.5):
    """Global SSIM as computed by MATLAB ``ssim`` (Gaussian window sigma 1.5,
    11x11, replicate padding, mean over the whole SSIM map)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    rad = int(np.ceil(3 * sigma))

    def f(x):
        return ndi.gaussian_filter(x, sigma, radius=rad, mode="nearest")

    mux, muy = f(a), f(b)
    sxx = f(a * a) - mux ** 2
    syy = f(b * b) - muy ** 2
    sxy = f(a * b) - mux * muy
    num = (2 * mux * muy + C1) * (2 * sxy + C2)
    den = (mux ** 2 + muy ** 2 + C1) * (sxx + syy + C2)
    return float(np.mean(num / den))


def _norm2(x):
    """MATLAB ``norm(matrix)``: largest singular value."""
    x = np.asarray(x, dtype=np.float64)
    if min(x.shape) < 3:
        return float(np.linalg.norm(x, 2))
    from scipy.sparse.linalg import svds
    try:
        return float(svds(x, k=1, return_singular_vectors=False)[0])
    except Exception:
        return float(np.linalg.norm(x, 2))


def mse(a, b):
    """Mean squared error (``wh_mse``)."""
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def sharpness(stack):
    """Mean Sobel gradient magnitude of each image (``rg_sharpness``)."""
    s = np.asarray(stack.data if isinstance(stack, TiltSeries) else stack)
    if s.ndim == 2:
        s = s[None]
    return np.array([imgradient_sobel(im).mean() for im in s])


# =========================================================================== robust fit
def robustfit(X, y, tune=4.685):
    """MATLAB ``robustfit(X, y)`` with bisquare weights (constant term added
    as the first coefficient)."""
    import scipy.linalg as sla
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    X = np.column_stack([np.ones(n), X])
    p = X.shape[1]
    eps = np.finfo(np.float64).eps

    Q, R, perm = sla.qr(X, mode="economic", pivoting=True)
    tol = abs(R[0, 0]) * max(n, p) * eps
    xrank = int(np.sum(np.abs(np.diag(R)) > tol))
    b = np.zeros(p)
    perm = perm[:xrank]
    Rk = R[:xrank, :xrank]
    b[perm] = sla.solve_triangular(Rk, Q[:, :xrank].T @ y)
    b0 = np.zeros(p)

    # leverage adjustment (DuMouchel & O'Brien)
    E = sla.solve_triangular(Rk.T, X[:, perm].T, lower=True).T
    h = np.minimum(0.9999, np.sum(E * E, axis=1))
    adjfactor = 1.0 / np.sqrt(1 - h)
    tiny_s = 1e-6 * np.std(y, ddof=1)
    if tiny_s == 0:
        tiny_s = 1.0
    D = np.sqrt(eps)
    wxrank = xrank
    it = 0
    while it == 0 or np.any(np.abs(b - b0) > D * np.maximum(np.abs(b), np.abs(b0))):
        it += 1
        if it > 50:
            import warnings
            warnings.warn("robustfit: iteration limit reached")
            break
        r = y - X @ b
        radj = r * adjfactor
        rs = np.sort(np.abs(radj))
        s = np.median(rs[max(1, wxrank) - 1:]) / 0.6745
        u = radj / (max(s, tiny_s) * tune)
        w = (np.abs(u) < 1) * (1 - u ** 2) ** 2
        b0 = b.copy()
        sw = np.sqrt(w)
        xw = X[:, perm] * sw[:, None]
        Qw, Rw = np.linalg.qr(xw)
        bb = np.linalg.lstsq(Rw, Qw.T @ (y * sw), rcond=None)[0]
        b[perm] = bb
        dR = np.abs(np.diag(Rw))
        # MATLAB wfit: sum(abs(diag(R)) > max(r,c)*eps(R(1))*abs(R(1)))
        wxrank = int(np.sum(dR > max(xw.shape) * np.spacing(dR[0]) * dR[0])) if dR.size else 0
    return b


# =========================================================================== Ajinkya's method
def _create_ref_stack(sino, angles):
    print("Generating reference stack ...")
    stack = TiltSeries(data=rescale(sino), angles=angles)
    rec = reconstruct(stack, "em", 5)
    return fp(rec, angles).astype(np.float32)


def _compute_ssim(sino, sino_ref):
    from tqdm.auto import tqdm
    print("Computing Structural Similarity Index...")
    out = np.zeros(sino.shape[0])
    for i in tqdm(range(sino.shape[0]), desc="SSIM", leave=False):
        xi = sino[i] / _norm2(sino[i])
        yi = sino_ref[i] / _norm2(sino_ref[i])
        out[i] = ssim(xi, yi)
    return out


def _remove_bad_images(ssim_x, angles):
    print("Removing distorted images from tilt_series...")
    x = np.asarray(angles, dtype=float)
    nA = x.size
    X = np.column_stack([x, x ** 2, x ** 3, x ** 4])
    b = robustfit(X, ssim_x)
    y = X @ b[1:] + b[0]
    tol, tol_std = 1e-4, 1.0
    min_images = max(80, int(np.floor(0.33 * nA))) if nA > 80 else nA # At least 80, 1/3 of acquisitions if it's more
    cXm = np.array([], dtype=int)
    for _ in range(100):
        if cXm.size and np.unique(x[cXm]).size >= min_images:
            break
        cXId = np.flatnonzero(np.abs(ssim_x - y) < tol * y)
        cXIds = np.flatnonzero(np.abs(ssim_x[cXId] - np.mean(y)) < tol_std * np.std(y, ddof=1))
        cXp = cXId[cXIds]
        best = []
        for ang in np.unique(x[cXp]):
            ang_i = cXp[x[cXp] == ang]
            best.append(ang_i[np.argmin(np.abs(ssim_x[ang_i] - y[ang_i]))])
        cXm = np.asarray(best, dtype=int)
        tol *= 1.2
        tol_std *= 2
    print(f"** Found {cXm.size} undistorted images out of {nA}")
    return cXm, y


def generate_clean_tilt_series(series, plot=False, save_dir=None):
    """Remove distorted projections by comparison with reprojections
    (``generateCleanTiltSeries``, recommended).

    The series is reconstructed (EM, 5 iterations) and re-projected; the
    SSIM between each projection and its re-projection is fitted with a robust
    4th-order polynomial of the tilt angle, and at each angle the projection
    closest to the fit is kept (at least 70 images, or all angles for short
    series). The returned data is the globally rescaled input.

    Parameters
    ----------
    plot : bool
        Show the SSIM, the robust fit and the selected projections.
    save_dir : str or None
        If given, also save the clean series as ``clean_series.npz`` there
        (the MATLAB version always saved ``clean_series.mat`` in the current
        folder).
    """
    sino = np.nan_to_num(np.asarray(series.data, dtype=np.float32))
    sino = rescale(sino)
    angles = np.asarray(series.angles, dtype=float)
    sino_ref = _create_ref_stack(sino, angles)
    ssim_x = _compute_ssim(sino, sino_ref)
    cXm, yfit = _remove_bad_images(ssim_x, angles)
    print("Generating clean series ...")
    out = TiltSeries(data=np.ascontiguousarray(sino[cXm]), angles=angles[cXm],
                     pixel_size=series.pixel_size)
    out.meta["kept_indices"] = cXm
    out.meta["ssim"] = ssim_x
    if save_dir is not None:
        import os
        np.savez(os.path.join(save_dir, "clean_series.npz"), data=out.data, angles=out.angles,
                 pixel_size=out.pixel_size)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.5))
        o = np.argsort(angles, kind="stable")
        ax.semilogy(angles[o], ssim_x[o], ".", color="0.5", label="SSIM (all projections)")
        ax.semilogy(angles[o], yfit[o], "-", lw=2, label="robust fit")
        ax.semilogy(angles[o], (yfit * (1 - 1.2e-4))[o], ":", color="C1")
        ax.semilogy(angles[o], (yfit * (1 + 1.2e-4))[o], ":", color="C1")
        ax.semilogy(angles[cXm], ssim_x[cXm], "o", mfc="none", color="C2", label="kept")
        ax.set(xlabel="Angle [deg]", ylabel="Structural similarity (SSIM)",
               title=f"Robust fitting (selected {cXm.size} out of {angles.size})")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()
    return out


# =========================================================================== Wouter's method
def generate_clean_series(series, retain_at_least=100, acceptance_multiplier=2.0, plot=True):
    """Remove distorted frames using the mean squared error to the neighbouring
    frames (``rg_generate_clean_series``, W. Heyvaert).

    Frames with an error above ``acceptance_multiplier`` times the
    ``retain_at_least``-th smallest error are removed. The kept stack is then
    centred (centre of mass) and only one frame per angle is kept (the
    sharpest)."""
    data = np.asarray(series.data, dtype=np.float32)
    angles = np.asarray(series.angles, dtype=float)
    n = data.shape[0]
    if n <= retain_at_least:
        print("There are fewer than RETAIN_AT_LEAST images, no images were removed.")
        return series.copy()
    err = np.zeros(n)
    err[0] = mse(data[0], data[1])
    err[-1] = mse(data[-1], data[-2])
    for i in range(1, n - 1):
        err[i] = 0.5 * (mse(data[i], data[i - 1]) + mse(data[i], data[i + 1]))
    t = acceptance_multiplier * np.sort(err)[retain_at_least - 1]
    keep = err < t
    out = TiltSeries(data=data[keep], angles=angles[keep], pixel_size=series.pixel_size)
    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(8, 5))
        axs[0].plot(err, "ko", ms=3)
        axs[0].plot(0 * err + t, "r-")
        axs[0].set_title("Selection criterium")
        axs[1].plot(out.angles, "k.")
        axs[1].set_title("Projection angles")
        fig.tight_layout()
        plt.show()
    out.data = align_proj_com(out.data)
    return fasttomo_keepone(out, "sharpest")


def fasttomo_keepone(series, method="first"):
    """Keep a single projection per tilt angle (``rg_fasttomo_keepone``).

    method: ``'first'``, ``'sharpest'`` (highest mean gradient) or
    ``'average'``. The result is sorted by angle.

    Note: the MATLAB loop only ever processed the first angle (``1:size(C)``),
    so it effectively always kept the first projection; here the documented
    behaviour is implemented."""
    data = np.asarray(series.data)
    angles = np.asarray(series.angles, dtype=float)
    C, ia, ic = np.unique(angles, return_index=True, return_inverse=True)
    out = np.empty((C.size,) + data.shape[1:], dtype=np.float32)
    for k in range(C.size):
        idx = np.flatnonzero(ic == k)
        if method == "first" or idx.size == 1:
            out[k] = data[ia[k]]
        elif method == "sharpest":
            out[k] = data[idx[np.argmax(sharpness(data[idx]))]]
        elif method == "average":
            out[k] = data[idx].mean(axis=0)
        else:
            raise ValueError(f"Unknown method '{method}'")
    if method == "first":
        print("Kept the first projection at every angle")
    return TiltSeries(data=out, angles=C, pixel_size=series.pixel_size)


# =========================================================================== manual
def select_images(series, exclude=None):
    """Remove the projections whose indices (0-based, as shown by the
    stackview sliders) are listed in ``exclude`` (``select_images_gui_2`` with
    a list). For the interactive version see :func:`rgtomo.select_images_gui`."""
    exclude = [] if exclude is None else list(np.atleast_1d(exclude).astype(int))
    keep = np.setdiff1d(np.arange(series.n_angles), exclude)
    out = series.select(keep)
    out.meta["excluded_indices"] = sorted(set(exclude))
    return out
