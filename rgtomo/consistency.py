"""Tomographic self-consistency alignment.

Port of ``align_tomo_consistency.m`` (R. Girod's TEM adaptation) and of the
cSAXS ``align_tomo_consistency_linear.m`` as used with the stock ASTRA
toolbox (parallel-beam geometry without detector offsets), i.e. the version
in rg-tomo-matlab. The projections are iteratively reconstructed (FBP),
re-projected and shifted so that they agree with the re-projections
(optical-flow / gradient-based shift estimate), first on 4x then on 2x binned
data.

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

Adapted for TEM by R. Girod (UAntwerp); Python port 2026.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import fft as sfft

from ._matlab import conv_same, gausswin, hanning, iround, quantile, tukeywin
from ._plotting import LiveFigure
from .csaxs import (center_of_mass, circular_mask, get_img_grad,
                    get_phase_gradient_1d, imfilter_high_pass_1d,
                    imgaussfilt3_conv, imrotate_ax_fft, imshift_fft,
                    imshift_linear, interpolate_ft_centered,
                    interpolate_linear, smooth_edges, unwrap2d_fft)
from .series import TiltSeries

__all__ = ["align_tomo_consistency"]

_W = -1


def _nextpow2(n):
    return int(np.ceil(np.log2(abs(n)))) if n > 0 else 0


# =========================================================================== public
def align_tomo_consistency(series, vert_range=None, high_pass_filter=0.01, align_horizontal=True,
                           align_vertical=False, refine_geometry=False, *, binning=(4, 2),
                           min_step_size=0.02, max_iter=500, momentum_acceleration=True,
                           apply_positivity=True, apply_geometry=False, plot=True,
                           plot_every=10.0, verbose=True):
    """Align a tilt series using tomographic self-consistency (reprojection).

    Parameters
    ----------
    series : TiltSeries
        Tilt series (bright object on dark background).
    vert_range : (start, stop) or None
        Rows (0-based, stop excluded) used for the alignment, e.g. to focus on
        a region of interest. None uses all rows.
    high_pass_filter : float
        Band-pass filter applied to the data, 1 = no filter, typically 0.01,
        lower values filter more (0.005 for STEM).
    align_horizontal, align_vertical : bool
        Refine the horizontal and/or vertical shifts.
    refine_geometry : bool
        Also refine the in-plane rotation (tilt) of each projection.
    binning : sequence of int
        Downsampling levels, coarse to fine (MATLAB: 4 then 2).
    min_step_size : float
        Stopping criterion (px, at full resolution).
    max_iter : int
        Maximum number of iterations per binning level.
    apply_geometry : bool
        If ``refine_geometry``, also rotate the output projections by the
        refined tilt (the MATLAB version only applies the shifts).
    plot : bool
        Show the progress figure (refreshed every ``plot_every`` seconds) and
        the final shifts.

    Returns
    -------
    TiltSeries
        The input projections shifted (Fourier shift, periodic boundaries) by
        the optimal shifts, stored in ``meta['consistency_shift']`` as [x, y].
    """
    data = np.asarray(series.data, dtype=np.float32)
    stack_object = -(data - data.max())          # needed by the internal cSAXS algorithms
    theta = np.asarray(series.angles, dtype=float)
    n_angles, n_rows, n_cols = data.shape

    width_sinogram = int(np.ceil(n_cols / 32) * 32)
    npix = int(np.ceil(width_sinogram / 32) * 32)
    par = dict(
        air_gap=np.array([10.0, 10.0]), high_pass_filter=high_pass_filter,
        momentum_acceleration=momentum_acceleration, apply_positivity=apply_positivity,
        valid_angles=np.ones(n_angles, dtype=bool), center_reconstruction=True,
        align_horizontal=bool(align_horizontal), align_vertical=bool(align_vertical),
        min_step_size=min_step_size, max_iter=int(max_iter), refine_geometry=bool(refine_geometry),
        step_relaxation=0.5, filter_type="ram-lak", freq_scale=1.0, unwrap_data_method="fft_1d",
        lamino_angle=90.0, tilt_angle=0.0, skewness_angle=0.0, verbose=verbose,
        plot=plot, plot_every=plot_every,
        vert_range=None if vert_range is None else [int(vert_range[0]) + 1, int(vert_range[1])],
    )

    if verbose:
        print("Get phase gradient")
    sinogram = get_phase_gradient_1d(stack_object, axis=-1)
    if verbose:
        print("Get mask")
    # include the effect of high pass filter into the weights; relevance
    # weights remove the effect of potential residues / phase jumps
    ker_len = max(3, int(np.ceil(high_pass_filter * width_sinogram)))
    jumps = (np.abs(sinogram) > 2).astype(np.float32)
    jumps = conv_same(conv_same(jumps, hanning(3), axis=1), gausswin(ker_len), axis=2)
    sino_weights = np.maximum(0, 1 - jumps).astype(np.float32)

    shift = np.zeros((n_angles, 2))
    live = LiveFigure() if plot else None
    tilt = np.zeros(n_angles)
    rec = None
    for b in binning:
        if verbose:
            print(f"Binning {b}")
        shift, tilt, rec, err = _align_tomo_consistency_linear(
            sinogram, sino_weights, theta + 0.1, npix, shift, par, int(b), live)
    if verbose:
        print("Alignment finished")
    if plot:
        _plot_total_shift(theta, shift, tilt if refine_geometry else None)

    aligned = imshift_fft(data, shift[:, 0], shift[:, 1]).astype(np.float32)
    if refine_geometry and apply_geometry and np.any(np.abs(tilt) > 1e-5):
        aligned = imrotate_ax_fft(aligned, -tilt).astype(np.float32)
    out = series.with_data(aligned)
    out.meta["consistency_shift"] = shift
    if refine_geometry:
        out.meta["consistency_tilt"] = tilt
    return out


# =========================================================================== helpers
def _imshift_generic(img, shift, smooth, roi_rows, downsample, method, interp_sign):
    """Port of ``imshift_generic``: shift, crop rows, smooth + downsample."""
    is_real = np.isrealobj(img)
    shift = np.asarray(shift, dtype=float)
    if np.any(shift != 0):
        nz = np.any(shift != 0, axis=0)
        axes = [ax for ax, flag in zip((-1, -2), nz) if flag]
        img = smooth_edges(img, smooth, axes=axes)
        if method == "linear":
            img = imshift_linear(img, shift[:, 0], shift[:, 1])
        else:
            img = imshift_fft(img, shift[:, 0], shift[:, 1])
    img = img[:, roi_rows, :]
    np_ = img.shape[1:]
    if downsample > 1:
        img = imgaussfilt3_conv(img, [0, downsample, downsample])
        img = img / imgaussfilt3_conv(np.ones(np_, dtype=img.dtype), [downsample, downsample])
        new = (int(np.ceil(np_[0] / downsample / 2) * 2), int(np.ceil(np_[1] / downsample / 2) * 2))
        if method == "linear":
            img = interpolate_linear(img, new)
        else:
            img = interpolate_ft_centered(smooth_edges(img, 2 * downsample), new, interp_sign)
    if is_real:
        img = np.real(img)
    return np.ascontiguousarray(img, dtype=np.float32)


def _fbp_filter(angles_deg, nw, filter_type="ram-lak", d=1.0):
    """FBP filter of cSAXS ``FBP.m`` including the weights for irregular
    angular sampling: returns (H (n_angles, order), order)."""
    nproj = len(angles_deg)
    theta = np.mod(np.deg2rad(angles_deg), 2 * np.pi)
    th = np.mod(theta - theta[0], np.pi)
    srt = np.argsort(th, kind="stable")
    ts = th[srt]
    w = np.zeros(nproj)
    if nproj > 2:
        w[1:-1] = (ts[2:] - ts[:-2]) / 2
    w[0] = ts[1] - ts[0] if nproj > 1 else 1
    w[-1] = ts[-1] - ts[-2] if nproj > 1 else 1
    weights = np.empty(nproj)
    weights[srt] = w
    med = np.median(weights)
    weights[weights > 2 * med] = med
    weights = weights / weights.mean()
    weights = weights * (np.pi / 2 / nproj)          # * sin(lamino_angle = 90 deg)

    order = max(64, 2 ** _nextpow2(2 * nw))
    k = np.arange(order // 2 + 1)
    filt = 2 * k / order
    wf = 2 * np.pi * k / order
    if filter_type == "shepp-logan":
        filt[1:] *= np.sin(wf[1:] / (2 * d)) / (wf[1:] / (2 * d))
    elif filter_type == "cosine":
        filt[1:] *= np.cos(wf[1:] / (2 * d))
    elif filter_type == "hamming":
        filt[1:] *= 0.54 + 0.46 * np.cos(wf[1:] / d)
    elif filter_type == "hann":
        filt[1:] *= (1 + np.cos(wf[1:] / d)) / 2
    elif filter_type != "ram-lak":
        raise ValueError(f"Invalid filter '{filter_type}'")
    filt[wf > np.pi * d] = 0
    filt = np.concatenate([filt, filt[-2:0:-1]])
    return (weights[:, None] * filt[None, :]).astype(np.float32), order


class _Projector:
    """ASTRA parallel-beam 3D projector on preallocated arrays (same geometry
    as the MATLAB ``ASTRA_GPU_wrapper`` with stock ASTRA)."""

    def __init__(self, n_layers, width, npix, angles_deg):
        import astra
        import astra.experimental  # noqa: F401
        self.astra = astra
        self.proj_geom = astra.create_proj_geom("parallel3d", 1.0, 1.0, n_layers, width,
                                                np.deg2rad(angles_deg))
        self.vol_geom = astra.create_vol_geom(npix, npix, n_layers)
        self.pid = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom)
        self.vol = np.zeros((n_layers, npix, npix), dtype=np.float32)
        self.proj = np.zeros((n_layers, len(angles_deg), width), dtype=np.float32)

    def bp(self, sino):
        """sino (n_angles, n_layers, width) -> volume (n_layers, npix, npix)."""
        self.proj[...] = np.transpose(sino, (1, 0, 2))
        self.astra.experimental.direct_BP3D(self.pid, self.vol, self.proj)
        return self.vol.copy()

    def fp(self, vol):
        """volume (n_layers, npix, npix) -> projections (n_angles, n_layers, width)."""
        self.vol[...] = vol
        self.astra.experimental.direct_FP3D(self.pid, self.vol, self.proj)
        return np.ascontiguousarray(np.transpose(self.proj, (1, 0, 2)))

    def close(self):
        self.astra.projector3d.delete(self.pid)


def _fbp(proj, sinogram, H, order):
    """Filtered back-projection (``FBP_zsplit``/``FBP``, zero padding)."""
    nw = sinogram.shape[-1]
    p = (order - nw) // 2
    s = np.pad(sinogram, [(0, 0), (0, 0), (p, p)])
    s = sfft.ifft(sfft.fft(s, axis=-1, workers=_W) * H[:, None, :], axis=-1, workers=_W).real
    s = s[..., order // 2 - nw // 2: order // 2 + nw // 2]
    return proj.bp(s.astype(np.float32))


def _get_img_grad_filtered(img, axis, hpf, smooth_win):
    """Filtered image gradient along x (``axis='x'``) or y (``axis='y'``)."""
    if axis == "x":
        img = smooth_edges(img, smooth_win, axes=[-1])
        n = img.shape[-1]
        X = (2j * np.pi * (np.fft.fftshift(np.arange(n) / n) - 0.5)).astype(np.complex64)
        d = sfft.fft(img, axis=-1, workers=_W) * X
        d = imfilter_high_pass_1d(d, -1, hpf, 0, apply_fft=False)
        d = sfft.ifft(d, axis=-1, workers=_W)
    else:
        img = smooth_edges(img, smooth_win, axes=[-2])
        n = img.shape[-2]
        X = (2j * np.pi * (np.fft.fftshift(np.arange(n) / n) - 0.5)).astype(np.complex64)
        d = sfft.fft2(img, axes=(-2, -1), workers=_W) * X[:, None]
        d = imfilter_high_pass_1d(d, -1, hpf, 0, apply_fft=False)
        d = sfft.ifft2(d, axes=(-2, -1), workers=_W)
    return d.real.astype(np.float32)


def _resid_sino(model, sino, hpf):
    return imfilter_high_pass_1d(model - sino, -1, hpf)


def _find_optimal_shift(model, sino, weights, mass, par):
    hpf = par["high_pass_filter"]
    n = model.shape[0]
    shift_x = np.zeros(n)
    shift_y = np.zeros(n)
    resid = _resid_sino(model, sino, hpf)
    if par["align_horizontal"]:
        dX = _get_img_grad_filtered(model, "x", hpf, 5)
        shift_x = -np.sum(weights * dX * resid, axis=(1, 2)) / np.sum(weights * dX ** 2, axis=(1, 2))
    if par["align_vertical"]:
        dY = _get_img_grad_filtered(model, "y", hpf, 5)
        shift_y = -np.sum(weights * dY * resid, axis=(1, 2)) / np.sum(weights * dY ** 2, axis=(1, 2))
    shift = np.column_stack([shift_x, shift_y])
    if np.any(np.isnan(shift)):
        import warnings
        warnings.warn("Alignment failed, estimated shift is NaN")
        shift = np.nan_to_num(shift)
    err = np.sqrt(np.mean((weights * resid) ** 2, axis=(1, 2))) / mass
    return shift, err


def _refine_tilt(model, measured, weights, hpf, tilt):
    """Gradient-descent update of the per-projection in-plane rotation
    (``refine_geometry``, 'tilt_angle' only)."""
    resid = _resid_sino(model, measured, hpf)
    dX, dY = get_img_grad(model)
    ny, nx = model.shape[-2:]
    D = dX * np.linspace(-1, 1, ny)[:, None] - dY * np.linspace(-1, 1, nx)[None, :]
    D = imfilter_high_pass_1d(D, -1, hpf)
    upd = np.sum(weights * resid * D, axis=(1, 2)) / np.sum(weights * D ** 2, axis=(1, 2))
    return tilt + 0.01 * np.rad2deg(upd)


def _add_momentum(shifts_memory, velocity, acc_axes):
    """Accelerated momentum gradient descent (``add_momentum``)."""
    from scipy.optimize import minimize
    shift = shifts_memory[-1].copy()
    if not np.any(acc_axes & np.any(shift != 0, axis=0)):
        return shift, velocity
    nmem = shifts_memory.shape[0] - 1
    lags = np.arange(nmem, 0, -1)
    for jj in np.flatnonzero(acc_axes):
        if np.all(shift[:, jj] == 0):
            continue
        C = np.array([np.corrcoef(shift[:, jj], shifts_memory[i, :, jj])[0, 1] for i in range(nmem)])
        if not np.all(np.isfinite(C)):
            decay = 0.0
        else:
            res = minimize(lambda x: np.linalg.norm(C - np.exp(-x[0] * lags)), x0=[0.0],
                           method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4})
            decay = float(res.x[0])
        alpha, gain = 2.0, 0.5
        friction = min(1.0, max(0.0, alpha * decay))
        velocity[:, jj] = (1 - friction) * velocity[:, jj] + shift[:, jj]
        shift[:, jj] = (1 - gain) * shift[:, jj] + gain * velocity[:, jj]
    return shift, velocity


def _align_tomo_consistency_linear(sinogram, weights, angles, npix, optimal_shift, par, binning, live):
    """Port of ``align_tomo_consistency_linear`` for standard tomography."""
    from tqdm.auto import tqdm
    n_angles = len(angles)
    n_layers0 = sinogram.shape[1]
    valid = par["valid_angles"]
    verbose = par["verbose"]
    hpf = par["high_pass_filter"]

    # ---------------- vertical range (1-based inclusive like MATLAB)
    vr = par["vert_range"] or [1, n_layers0]
    vr0 = [max(vr[0], iround(1 + optimal_shift[:, 1].max())),
           min(vr[-1], int(np.floor(n_layers0 + optimal_shift[:, 1].min())))]
    vrange = [max(1, vr0[0]), min(vr0[1], n_layers0)]
    center = int(np.ceil(np.mean(vrange)))
    split = 2 ** _nextpow2(np.ceil(npix * npix * n_layers0 * 8 / 1e9 / binning ** 3) * binning)
    nvert = int(np.floor((vrange[1] - vrange[0] + 1) / split) * split)
    vrange = [int(np.ceil(center - nvert / 2)), int(np.floor(center + nvert / 2 - 1))]
    if vrange != vr0 and verbose:
        print(f"Changing vertical range from {vr0[0]}:{vr0[1]} to {vrange[0]}:{vrange[1]}")
    if vrange[1] < vrange[0] or (par["align_vertical"] and vrange[1] - vrange[0] + 1 < 10 * binning):
        raise ValueError("Too small vertical range, extend the alignment range")
    roi = slice(vrange[0] - 1, vrange[1])

    if np.all(weights == 1):
        weights = None
    if verbose:
        print(f"Shifting sinograms and binning = {binning}")
    interp_sign = 1    # 'fft_1d' unwrapping
    sino = _imshift_generic(sinogram, optimal_shift, 5, roi, binning, "fft", interp_sign)
    if weights is not None:
        weights = _imshift_generic(weights, optimal_shift, 0, roi, binning, "linear", interp_sign)

    # ---------------- sort by angle
    ang_order = np.argsort(angles, kind="stable")
    sino = sino[ang_order]
    if weights is not None:
        weights = weights[ang_order]
    angles = angles[ang_order]
    opt_sorted = optimal_shift[ang_order]
    valid = valid[ang_order]

    n_layers, width = sino.shape[1:]
    win = tukeywin(width, 0.2)[None, None, :]
    if n_layers > 10 and par["align_vertical"]:
        win = tukeywin(n_layers, 0.2)[None, :, None] * win
    win = win.astype(np.float32)

    max_iter = par["max_iter"]
    shift_upd_all = np.full((max_iter, n_angles, 2), np.nan)
    shift_upd_all[0] = 0
    velocity = np.zeros((n_angles, 2))
    shift_total = np.zeros((n_angles, 2))
    err_all = []
    npix_b = int(np.ceil(npix / binning))
    circulo = circular_mask(npix_b, 5)
    w_static = 1.0 if weights is None else None
    tilt = np.zeros(n_angles)
    geometry_hist = [0.0]

    H, order = _fbp_filter(angles, width, par["filter_type"], par["freq_scale"])
    proj = _Projector(n_layers, width, npix_b, angles)
    air_gap = par["air_gap"] / binning
    mass_ref = None
    rec_center_0 = np.zeros(2)
    t_plot = time.time()
    last_round = False
    rec = None
    pbar = tqdm(range(max_iter), desc=f"Consistency alignment (binning {binning})", leave=True,
                disable=not verbose)
    try:
        for ii in pbar:
            # -------- shift (and rotate) the data
            sino_shifted = imshift_fft(sino, shift_total[:, 0], shift_total[:, 1])
            if np.any(np.abs(tilt) > 1e-5):
                sino_shifted = imrotate_ax_fft(sino_shifted, -tilt)
            if weights is not None:
                w = imshift_linear(weights, shift_total[:, 0], shift_total[:, 1])
                w = np.maximum(0, w * win)
            else:
                w = np.maximum(0, w_static * win)
            sino_shifted = -unwrap2d_fft(sino_shifted, axis=-1, empty_region=air_gap).astype(np.float32)
            if mass_ref is None:
                mass_ref = float(np.median(np.mean(np.abs(sino_shifted), axis=(1, 2))))

            # -------- FBP + prior knowledge
            rec = _fbp(proj, sino_shifted, H, order)
            rec *= circulo
            if par["apply_positivity"]:
                rec = np.maximum(0, rec)

            # -------- keep the reconstruction centred
            if par["align_horizontal"]:
                x, y, mass = center_of_mass(np.sqrt(np.maximum(0, rec)) + np.finfo(np.float64).eps)
                if n_layers > 5:
                    mass = mass * tukeywin(n_layers, 0.1)
                rec_center = np.array([np.mean(x * mass) / np.mean(mass), np.mean(y * mass) / np.mean(mass)])
                shift_rec = -0.5 * (rec_center - rec_center_0)
                rec = imshift_fft(rec, shift_rec[0], shift_rec[1]).astype(np.float32)

            # -------- reprojection and shift estimate
            model = proj.fp(rec)
            if par["refine_geometry"]:
                tilt = _refine_tilt(model, sino_shifted, w, hpf, tilt)
                geometry_hist.append(float(np.mean(tilt)))
            shift_upd, err = _find_optimal_shift(model, sino_shifted, w, mass_ref, par)
            err_all.append(err)
            shift_upd = np.minimum(0.5, np.abs(shift_upd)) * np.sign(shift_upd) * par["step_relaxation"]
            shift_upd_all[ii] = shift_upd
            if par["momentum_acceleration"]:
                max_update = quantile(np.abs(shift_upd[valid]), 0.995, axis=0)
                if ii > 1:
                    shift_upd, velocity = _add_momentum(shift_upd_all[ii - 2:ii + 1], velocity,
                                                        max_update * binning < 0.5)
            shift_upd[:, 1] -= np.median(shift_upd[:, 1])
            max_step = np.minimum(quantile(np.abs(shift_upd), 0.99, axis=0), 0.5)
            shift_upd = np.minimum(max_step, np.abs(shift_upd)) * np.sign(shift_upd)
            shift_total = shift_total + shift_upd

            max_update = float(np.max(quantile(np.abs(shift_upd[valid]), 0.995, axis=0)))
            pbar.set_postfix(err=f"{np.mean(err):.4g}", step=f"{max_update * binning:.3g}px",
                             stop=f"{par['min_step_size']}px")
            if max_update * binning < par["min_step_size"]:
                last_round = True
            if live is not None and (time.time() - t_plot > par["plot_every"] or last_round
                                     or ii == max_iter - 1):
                _plot_alignment(live, rec, sino_shifted, np.array(err_all), shift_upd, shift_total,
                                angles, valid, binning, hpf, tilt if par["refine_geometry"] else None)
                t_plot = time.time()
            if last_round:
                break
    finally:
        proj.close()
        pbar.close()
    if verbose:
        print("Minimal step limit reached" if last_round else "Maximum number of iterations reached")

    shift_total[:, 1] -= np.median(shift_total[:, 1])
    out_shift = np.empty_like(optimal_shift)
    out_shift[ang_order] = opt_sorted + shift_total * binning
    out_tilt = np.empty(n_angles)
    out_tilt[ang_order] = tilt
    err_all = np.array(err_all)
    err_out = np.empty_like(err_all)
    if err_all.size:
        err_out[:, ang_order] = err_all
    return out_shift, out_tilt, rec, err_out


# =========================================================================== plots
def _plot_alignment(live, rec, sino_shifted, err, shift_upd, shift_total, angles, valid,
                    binning, hpf, tilt=None):
    import matplotlib.pyplot as plt
    nl = sino_shifted.shape[1]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    sl = imfilter_high_pass_1d(sino_shifted[:, nl // 2, :], -1, hpf)
    lim = quantile(sl, [0.01, 0.99])
    axs[0, 0].imshow(sl, cmap="bone", vmin=lim[0], vmax=lim[1], aspect="auto")
    axs[0, 0].set_title(f"High-pass filtered shifted sinogram\nCurrent downsampling: {binning}x")
    axs[0, 0].axis("off")
    axs[0, 1].plot(angles, shift_upd[:, 0] * binning, ".-r", label="horiz")
    axs[0, 1].plot(angles, shift_upd[:, 1] * binning, ".-b", label="vert")
    axs[0, 1].set(title="Current position update", ylabel="Shift x downsampling [px]", xlabel="Angle [deg]")
    axs[0, 1].legend(); axs[0, 1].grid(True)
    axs[0, 2].plot(angles, shift_total[:, 0] * binning, ".-r", label="horiz")
    axs[0, 2].plot(angles, shift_total[:, 1] * binning, ".-b", label="vert")
    if tilt is not None:
        axs[0, 2].plot(angles, tilt, ".-g", label="tilt [deg]")
    axs[0, 2].set(title="Total position update", ylabel="Shift x downsampling [px]", xlabel="Angle [deg]")
    axs[0, 2].legend(); axs[0, 2].grid(True)
    mid = rec[rec.shape[0] // 2]
    lim = quantile(mid, [0.01, 0.999])
    axs[1, 0].imshow(mid, cmap="bone", vmin=lim[0], vmax=max(lim[1], lim[0] + 1e-12))
    axs[1, 0].set(title="Current reconstruction", xticks=[], yticks=[], ylabel="Horizontal cut")
    if err.ndim == 2 and err.shape[0] > 0:
        it = np.arange(1, err.shape[0] + 1)
        axs[1, 1].loglog(it, err, lw=0.5)
        axs[1, 1].loglog(it, err.mean(axis=1), "k", lw=3)
        axs[1, 1].set(title="MSE evolution", xlabel="Iteration", ylabel="Mean square error")
        axs[1, 1].grid(True)
        axs[1, 2].plot(angles[valid], err[-1][valid], "k.", label="errors")
        if np.any(~valid):
            axs[1, 2].plot(angles[~valid], err[-1][~valid], "r.", label="ignored")
            axs[1, 2].legend()
        axs[1, 2].set(title="Current error", xlabel="Angle [deg]")
        axs[1, 2].grid(True)
    fig.tight_layout()
    live.update(fig)


def _plot_total_shift(theta, shift, tilt=None):
    import matplotlib.pyplot as plt
    order = np.argsort(theta)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(theta[order], shift[order, 0], ".", label="Horizontal shift")
    ax.plot(theta[order], shift[order, 1], ".", label="Vertical shift")
    if tilt is not None:
        ax2 = ax.twinx()
        ax2.plot(theta[order], tilt[order], "g.", label="Refined tilt [deg]")
        ax2.set_ylabel("Tilt [deg]")
    ax.set(xlabel="Sorted angles", ylabel="Shift [px]",
           title="Total shift from self-consistency alignment")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    plt.show()
